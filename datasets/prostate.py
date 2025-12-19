# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from datasets import base

dataset_root = '/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/'
examples_train_path = 'imagesTr/'
labels_train_path = 'labelsTr/'
examples_test_path = 'imagesTs/'
labels_test_path = 'labelsTs/'

class Dataset(base.NiftiDataset):
    """Prostate"""

    def is_2d(self) -> bool: return True  # Prostate dataset is 3D

    @staticmethod
    def input_shape(): return (24, 320, 320) # (D, H, W) #TODO correct shape

    @staticmethod
    def num_train_examples(): return 600 # 25 files with 24 slices each = 600 slices

    @staticmethod
    def num_test_examples(): return 168 # 7 files with 24 slices each = 168 slices

    @staticmethod
    def num_classes(): return 2

    def __init__(self, examples_location: str, labels_location: str, is_training_set: bool):
        super(Dataset, self).__init__(examples_location, labels_location, is_training_set)

    @staticmethod
    def get_train_set(use_augmentation) -> 'Dataset':
        return Dataset(os.path.join(dataset_root, examples_train_path), os.path.join(dataset_root, labels_train_path), True)

    @staticmethod
    def get_test_set() -> 'Dataset':
        return Dataset(os.path.join(dataset_root, examples_test_path), os.path.join(dataset_root, labels_test_path), False)
    
DataLoader = base.DataLoader
