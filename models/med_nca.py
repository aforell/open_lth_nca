from foundations import hparams
from losses import loss_functions
from lottery.desc import LotteryDesc
from models import base
from models import nca
from pruning import sparse_global
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.image_utils as utils
from torchvision import transforms

_3D = '3d'
_2D = '2d'
_CLASSIFICATION = 'classification'
_SEGMENTATION = 'segmentation'

class Model(base.Model):
    '''An implamentation of a Med-NCA model'''
    '''Use it like this: nca_<2d/3d>_<segmentation/classification>'''

    def __init__(self, plan, initializer, model_hparams: hparams.ModelHparams, outputs=2):
        super(Model, self).__init__()
        self.state_channel_num = model_hparams.state_channel_num
        self.input_channel_num = model_hparams.input_channel_num
        self.use_patching = model_hparams.use_patching
        self.dimensions = plan[0]
        self.task = plan[1]
        
        self.scale_factor = model_hparams.scale_factor

        self.nca_1 = nca.Model(plan, initializer, model_hparams, outputs, res='low_res')
        self.nca_2 = nca.Model(plan, initializer, model_hparams, outputs, res='high_res')

        if self.task == _SEGMENTATION:
            self.criterion = loss_functions.DiceCELoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        
        if(self.task != _SEGMENTATION):
            raise NotImplementedError("Only segmentation task is implemented for Med-NCA")
        
        # Image padding to increase the number of channels
        CHANNEL_DIM = 1
        if x.shape[CHANNEL_DIM] < self.state_channel_num:
            pad_channels = self.state_channel_num - x.shape[CHANNEL_DIM]
            zeros = torch.zeros(x.shape[0], pad_channels, x.shape[2], x.shape[3], device=x.device) if self.dimensions == _2D else torch.zeros(x.shape[0], pad_channels, x.shape[2], x.shape[3], x.shape[4], device=x.device)
            x = torch.cat([x, zeros], dim=CHANNEL_DIM)
        
        x_scaled = F.interpolate(x, scale_factor=1/self.scale_factor, mode='trilinear' if self.dimensions == _3D else 'bilinear')
        x_scaled = self.nca_1(x_scaled)
        x_scaled = F.interpolate(x_scaled, scale_factor=self.scale_factor, mode='trilinear' if self.dimensions == _3D else 'bilinear')
        
        x = torch.cat((x[:, :self.input_channel_num, ...], x_scaled[:, self.input_channel_num:, ...]), CHANNEL_DIM) 

        if self.training and self.use_patching:
            if self.dimensions == _3D:
                raise NotImplementedError("3D random cropping not implemented yet")
            crop_params = transforms.RandomCrop.get_params(x, (x.shape[2]//self.scale_factor, x.shape[3]//self.scale_factor))
            patch = transforms.functional.crop(x, *crop_params)
            patch = self.nca_2(patch)

            loss_parameters = {'crop_params': crop_params}

            return (patch, loss_parameters)
        else:
            x = self.nca_2(x)
            return x

    @property
    def output_layer_names(self):
        #TODO importantly: what is this
        #TODO what is this
        if self.task == _CLASSIFICATION:
            return ['nca_2.classifier_fc1.weight', 'nca_2.classifier_fc1.bias']
        return ['nca_2.fc1.weight', 'nca_2.fc1.bias'] # TODO this can't be right as these layers are in the update function

    @staticmethod
    def is_valid_model_name(model_name):
        parts = model_name.split('_')
        if len(parts) >= 4:
            return parts[0] == 'med' and parts[1] == 'nca' and parts[2] in [_2D, _3D] and parts[3] in [_SEGMENTATION, _CLASSIFICATION]
        return False

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None, model_hparams=None):
        parts = model_name.split('_')
        plan = [parts[2], parts[3]]
        return Model(plan, initializer, model_hparams=model_hparams, outputs=outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='med_nca',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
            nca_steps_low_res=40,
            nca_steps_high_res=20,
            hidden_size=128,
            fire_rate=0.5,
            input_channel_num=1,
            state_channel_num=32,
            scale_factor=4,
            use_patching=False
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='prostate',
            batch_size=10
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=0.001,
            training_steps='2000ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global', #TODO check pruning
            pruning_fraction=0.2,
            #pruning_layers_to_ignore='fc.weight',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
