# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import nibabel as nib
import numpy as np
import einops
import torch
from datasets import base
import cv2

import utils.image_utils as utils

original_dataset_root = '/local/scratch/jkalkhof/Data/Prostate/Prostate_MEDSeg/'
preprocessed_dataset_root = '/local/scratch/aforell-thesis/open_lth_datasets/prostate2d/slices/'
examples_train_path = 'imagesTr/'
labels_train_path = 'labelsTr/'
examples_test_path = 'imagesTs/'
labels_test_path = 'labelsTs/'

class Dataset(base.Dataset):
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

    def __len__(self):
        return self.num_train_examples() if self._is_training_set else self.num_test_examples()

    def __init__(self, examples_location: str, labels_location: str, is_training_set: bool):
        self._is_training_set = is_training_set
        self._processed_examples_path = os.path.join(preprocessed_dataset_root, examples_location)
        self._processed_labels_path = os.path.join(preprocessed_dataset_root, labels_location)
        self._original_examples_path = os.path.join(original_dataset_root, examples_location)
        self._original_labels_path = os.path.join(original_dataset_root, labels_location)

        if not os.path.exists(self._processed_examples_path) or not os.path.exists(self._processed_labels_path):
            os.makedirs(preprocessed_dataset_root, exist_ok=True)
            os.makedirs(self._processed_examples_path)
            os.makedirs(self._processed_labels_path)
            for f in sorted([f for f in os.listdir(self._original_examples_path) if f.endswith('.nii') or f.endswith('.nii.gz')]):
                self.save_slices(f)
        self._example_files = sorted([ f for f in os.listdir(self._processed_examples_path) if f.endswith('.npy')])

    def __getitem__(self, index):
        example_path = os.path.join(self._processed_examples_path, self._example_files[index])
        label_path = os.path.join(self._processed_labels_path, self._example_files[index])
        example = np.load(example_path)
        label = np.load(label_path)

        example = torch.from_numpy(example)
        label = torch.from_numpy(label)
        return example, label

    def save_slices(self, file):
        print(f"Processing file: {file}")
        examples_path = os.path.join(self._original_examples_path, file)
        example = nib.load(examples_path)
        example = example.get_fdata().astype(np.float32) # h w d c
        example = einops.rearrange(example[:, :, :, 0], 'h w d -> d h w') #remove 4th dimension - as it is some second scan
        example = utils.rescale(img=example, goal_shape=self.input_shape(), is_label=False, is_3d=True)
        # TODO image preprocessing maybe here
        # Normalize image (optional)
        # example = (example - np.min(example)) / (np.max(example) - np.min(example) + 1e-8)
        example = einops.rearrange(example, 'd h w -> 1 d h w')  # add channel dim → (C, D, H, W)
    
        label_path = os.path.join(self._original_labels_path, file)
        label = nib.load(label_path).get_fdata().astype(np.uint8)
        label = einops.rearrange(label, 'h w d -> d h w')
        label = utils.rescale(img=label, goal_shape=self.input_shape(), is_label=True, is_3d=True)

        depth = example.shape[1]
        for i in range(depth):
            slice_example = example[:, i, :, :]  # (C, H, W)
            slice_label = label[i, :, :]
            slice_filename = file.replace('.nii.gz', f'_slice_{i}.npy')
            np.save(os.path.join(self._processed_examples_path, slice_filename), slice_example)
            np.save(os.path.join(self._processed_labels_path, slice_filename), slice_label)

    @staticmethod
    def get_train_set(use_augmentation) -> 'Dataset':
        return Dataset(examples_train_path, labels_train_path, True)

    @staticmethod
    def get_test_set() -> 'Dataset':
        return Dataset(examples_test_path, labels_test_path, False)
    
DataLoader = base.DataLoader
