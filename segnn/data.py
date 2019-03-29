import os
import glob

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def parse_id(path):
    return os.path.basename(path).split('.')[0]


def make_samples(data_dir):
    image_paths = sorted(glob.glob(os.path.join(data_dir, 'images/*.png')))
    label_paths = sorted(glob.glob(os.path.join(data_dir, 'labels/*.png')))
    if len(label_paths) == 0:
        label_paths = [None] * len(image_paths)
    samples = [*zip(image_paths, label_paths)]
    return samples


def random_resize(image, label):
    # np.random.randint right exclusive
    s = 0.5 + np.random.randint(0, 10) / 10
    image = cv2.resize(image, None, fx=s, fy=s,
                       interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, None, fx=s, fy=s,
                       interpolation=cv2.INTER_NEAREST)
    return image, label


def resize(image, label, size):
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if label is not None:
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
    return image, label


def random_crop(image, label, size):
    h, w, c = image.shape
    dh, dw = size
    i = np.random.randint(0, h - dh)
    j = np.random.randint(0, w - dw)
    image = image[i:i+dh, j:j+dw]
    label = label[i:i+dh, j:j+dw]
    return image, label


def random_resized_crop(image, label, size):
    image, label = random_resize(image, label)
    if image.shape < size:
        image, label = resize(image, label, size)
    image, label = random_crop(image, label, size)
    return image, label


class TrainDataset(Dataset):
    def __init__(self, data_dir, mean, tensor_size, resized=False, mirror=False):
        self.mean = mean
        self.tensor_size = tuple(tensor_size)
        self.resized = resized
        self.mirror = mirror
        self.samples = make_samples(data_dir)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]
        id_ = parse_id(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        size = np.array(image.shape[:2])  # original size

        if self.resized:
            image, label = random_resized_crop(image, label, self.tensor_size)
        else:
            image, label = random_crop(image, label, self.tensor_size)

        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, ::flip]
            label = label[:, ::flip]

        image = np.asarray(image, np.float32)
        image -= self.mean

        # h, w, c -> c, h, w
        image = image.transpose([2, 0, 1])

        return id_, size, image.copy(), label.copy()

    def __len__(self):
        return len(self.samples)


class TestDataset(Dataset):
    def __init__(self, data_dir, mean, tensor_size=None):
        self.mean = mean
        self.tensor_size = tuple(tensor_size)
        self.samples = make_samples(data_dir)

    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        id_ = parse_id(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        size = np.array(image.shape[:2])  # original size

        image, _ = resize(image, None, self.tensor_size)

        image = np.asarray(image, np.float32)
        image -= self.mean

        # h, w, c -> c, h, w
        image = image.transpose([2, 0, 1])
        image = torch.tensor(image)

        return id_, size, image, ''

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    ids = [sample[0] for sample in batch]
    sizes = [sample[1] for sample in batch]
    images = [sample[2] for sample in batch]
    labels = [sample[3] for sample in batch]
    return ids, sizes, images, labels


def make_train_dl(data_dir, batch_size, mean, tensor_size, resized=False, mirror=False, num_workers=8):
    dataset = TrainDataset(data_dir, mean,
                           tensor_size, resized, mirror)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers)

    return data_loader


def make_test_dl(data_dir, batch_size, mean, tensor_size, num_workers=8):
    dataset = TestDataset(data_dir, mean, tensor_size)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers)

    return data_loader
