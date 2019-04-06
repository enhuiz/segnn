import torch
import random
import numpy as np
import cv2
import math

from torchvision.transforms import Compose


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        img = np.array(img).astype(np.float32)
        img -= self.mean
        img /= self.std
        img /= 255

        sample['image'] = img
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()

        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.5, 1.0)):
        if size is not None:
            self.size = tuple(size)
        self.scale = scale

    def get_params(self, img):
        h, w, _ = img.shape
        ratio = random.uniform(*self.scale)
        dh = int(ratio * h)
        dw = int(ratio * w)
        i = random.randint(0, h - dh)
        j = random.randint(0, w - dw)
        return i, j, h, w

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        i, j, h, w = self.get_params(img)

        img = img[i:i+h, j:j+w]
        mask = mask[i:i+h, j:j+w]
        
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        sample['image'] = img
        sample['label'] = mask
        return sample


class Resize(object):
    def __init__(self, size):
        if size is not None:
            self.size = tuple(size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        sample['image'] = img
        sample['label'] = mask
        return sample
