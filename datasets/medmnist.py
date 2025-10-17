# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import torchvision
import medmnist

from datasets import base
from platforms.platform import get_platform


class Dataset(base.ImageDataset):
    """The MedMNIST dataset wrapper."""

    # Change this to the MedMNIST subclass you want, e.g. PathMNIST, OrganMNIST, etc.
    # Full list: https://medmnist.com/
    DATASET_CLASS = medmnist.PathMNIST  # Example: PathMNIST
    DATASET_NAME = "pathmnist"

    @staticmethod
    def num_train_examples():
        info = medmnist.INFO[Dataset.DATASET_NAME]
        return info["n_samples"]["train"]

    @staticmethod
    def num_test_examples():
        info = medmnist.INFO[Dataset.DATASET_NAME]
        return info["n_samples"]["test"]

    @staticmethod
    def num_classes():
        info = medmnist.INFO[Dataset.DATASET_NAME]
        return len(info["label"])

    @staticmethod
    def get_train_set(use_augmentation):
        train_set = Dataset.DATASET_CLASS(
            split="train",
            root=os.path.join(get_platform().dataset_root, "medmnist"),
            download=True,
        )
        info = medmnist.INFO[Dataset.DATASET_NAME]
        print(len(info["label"]))
        return Dataset(train_set.imgs, train_set.labels)

    @staticmethod
    def get_test_set():
        test_set = Dataset.DATASET_CLASS(
            split="test",
            root=os.path.join(get_platform().dataset_root, "medmnist"),
            download=True,
        )
        return Dataset(test_set.imgs, test_set.labels)

    def __init__(self, examples, labels):
        # Convert labels from [N,1] to [N]
        if len(labels.shape) > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        # Convert to tensor + normalize based on MedMNIST docs
        tensor_transforms = [
            torchvision.transforms.Normalize(mean=[.5], std=[.5])
        ]
        super(Dataset, self).__init__(examples, labels, [], tensor_transforms)

    def example_to_image(self, example):
        # MedMNIST images are numpy arrays (HxWxC)
        if example.ndim == 2:  # grayscale
            return Image.fromarray(example, mode="L")
        return Image.fromarray(example)


DataLoader = base.DataLoader
