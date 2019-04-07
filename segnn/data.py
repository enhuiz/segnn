import os
import glob

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import segnn.transforms as transforms


def parse_id(path):
    return os.path.basename(path).split('.')[0]


def make_samples(data_dir):
    image_paths = sorted(glob.glob(os.path.join(data_dir, 'images/*.png')))
    label_paths = sorted(glob.glob(os.path.join(data_dir, 'labels/*.png')))
    if len(label_paths) == 0:
        label_paths = [None] * len(image_paths)
    samples = [*zip(image_paths, label_paths)]
    return samples


class Task2Dataset(Dataset):
    def __init__(self, data_dir, mode, mean, input_size=None):
        self.mean = mean
        self.mode = mode
        self.input_size = input_size
        self.samples = make_samples(data_dir)

        self.train_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=self.mean, std=(1, 1, 1)),
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.Normalize(mean=self.mean, std=(1, 1, 1)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        id_ = parse_id(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if label_path:
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        else:
            label = np.zeros_like(image)

        shape = np.array(image.shape[:2])

        sample = {
            'id': id_,
            'image': image,
            'label': label,
            'shape': shape,  # height, width
        }

        if self.mode == 'train':
            sample = self.train_transform(sample)
        else:
            sample = self.test_transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
