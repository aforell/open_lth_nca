import cv2
import numpy as np
import torch
import torchvision


def show_image(image: torch.Tensor, title):
    """Display a given image.

    Args:
        image: A PIL Image or a torch Tensor.
        title: An optional title for the image.
    """
    import matplotlib.pyplot as plt

    images = []
    if (image.dim() == 3):
        for i in range(image.shape[0]):
          img_show = torchvision.transforms.ToPILImage(image[i, :, :])
          images.append(img_show)
          plt.imshow(img_show)

    plt.title(title)
    plt.legend()
    plt.savefig('images/' + title + '.png')

def rescale(img: np.ndarray, goal_shape, is_label: bool = False, is_3d: bool = False) -> np.ndarray:
        r"""Rescale input image to fit training size
            #Args
                img (numpy): Image data
                isLabel (numpy): Whether or not data is label
            #Returns:
                img (numpy): numpy array
        """
        if not is_3d:
            raise NotImplementedError("3D rescaling not implemented yet")

        interpolation = cv2.INTER_NEAREST if is_label else cv2.INTER_CUBIC
        shape = goal_shape
        img_resized = np.zeros(shape, dtype=img.dtype)
        for x in range(img.shape[0]):
            img_resized[x, :, :] = cv2.resize(img[x, :, :], dsize=(shape[2], shape[1]), interpolation=interpolation)
        
        img = img_resized
        img_resized = np.zeros(shape, dtype=img.dtype)
        for x in range(img.shape[2]):
            img_resized[:, :, x] = cv2.resize(img[:, :, x], dsize=(shape[1], shape[0]), interpolation=interpolation)

        return img_resized
