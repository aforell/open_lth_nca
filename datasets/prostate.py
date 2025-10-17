# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import concurrent
import nibabel as nib
import numpy as np
import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform

class Dataset(base.Dataset):
    """Prostate"""

    def getFilesInPath(self, path: str) -> dict:
        r"""Get files in path ordered by id and slice
            #Args
                path (string): The path which should be worked through
            #Returns:
                dic (dictionary): {key:patientID, {key:sliceID, img_slice}
        """
        dir_files = os.listdir(os.path.join(path))
        dic = {}
        for id_f, f in enumerate(dir_files):
            id = f
            if self.slice is not None:
                for slice in range(self.getSlicesOnAxis(os.path.join(path, f), self.slice)):
                    if id not in dic:
                        dic[id] = {}
                    dic[id][slice] = (f, id_f, slice)
        return dic

    def getSlicesOnAxis(self, path: str, axis: int) -> nib.nifti1:
            return self.load_item(path).shape[axis]

    def load_item(self, path: str) -> nib.nifti1:
            r"""Loads the data of an image of a given path.
                #Args
                    path (String): The path to the nib file to be loaded."""
            return nib.load(path).get_fdata()

    def __init__(self, loc: str, image_transforms):
        # Load the data.
        classes = sorted(get_platform().listdir(loc))
        samples = []

        for y_num, y_name in enumerate(classes):
            samples += _get_samples(loc, y_name, y_num)

        examples, labels = zip(*samples)
        super(Dataset, self).__init__(
            np.array(examples), np.array(labels), image_transforms,
            [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @staticmethod
    def num_train_examples(): return 1281167

    @staticmethod
    def num_test_examples(): return 50000

    @staticmethod
    def num_classes(): return 1000

    @staticmethod
    def _augment_transforms():
        return [
            torchvision.transforms.RandomResizedCrop(224, scale=(0.1, 1.0), ratio=(0.8, 1.25)),
            torchvision.transforms.RandomHorizontalFlip()
        ]

    @staticmethod
    def _transforms():
        return [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224)]

    @staticmethod
    def get_train_set(use_augmentation):
        transforms = Dataset._augment_transforms() if use_augmentation else Dataset._transforms()
        return Dataset(os.path.join(get_platform().imagenet_root, 'train'), transforms)

    @staticmethod
    def get_test_set():
        return Dataset(os.path.join(get_platform().imagenet_root, 'val'), Dataset._transforms())

    @staticmethod
    def example_to_image(example):
        with get_platform().open(example, 'rb') as fp:
            return Image.open(fp).convert('RGB')


DataLoader = base.DataLoader
