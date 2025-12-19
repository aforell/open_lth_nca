import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.image_utils as img_utils
from torchvision import transforms

def dice_score(logits, targets, smooth=1e-5):
    # logits: shape [N, C, H, W] or [N, C, D, H, W]
    logits = logits[:, -1, ...]
    probs = F.sigmoid(logits)
    probs = torch.flatten(probs)
    targets = torch.flatten(targets)
    targets = targets.clamp(0, 1) # simplify segmentation to binary task

    intersection = (probs * targets).sum()
    cardinality = (probs + targets).sum()
    dice_score = (2. * intersection + smooth) / (cardinality + smooth)
    return dice_score

class DiceCELoss(nn.Module):
    def __init__(self, smooth=1e-5, weight_ce=0.5, weight_dice=0.5):
        super(DiceCELoss, self).__init__()
        self.smooth = smooth
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, targets, loss_parameters=None):

        if loss_parameters is not None:
            crop_params = loss_parameters.get('crop_params', None)
            if crop_params is not None:
                targets = transforms.functional.crop(targets, *crop_params)

        # logits: shape [N, C, H, W] or [N, C, D, H, W]
        logits = logits[:, -1, ...]

        # Dice part
        probs = F.sigmoid(logits)
        probs = torch.flatten(probs)
        targets = torch.flatten(targets)
        targets = targets.clamp(0, 1) # simplify segmentation to binary task

        intersection = (probs * targets).sum()
        cardinality = (probs + targets).sum()
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score

        # Cross Entropy part
        loss_bce = torch.nn.functional.binary_cross_entropy(probs, targets.float(), reduction='mean')

        total_loss = self.weight_ce * loss_bce + self.weight_dice * dice_loss
        #print(f"Dice Loss: {dice_loss.item()}, BCE Loss: {loss_bce.item()}, Total Loss: {total_loss.item()}")
        return total_loss
