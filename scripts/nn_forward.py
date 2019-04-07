import os
import numpy as np
import argparse
import sys

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from segnn.data import Task2Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--model-path')
    parser.add_argument('--out-dir')
    parser.add_argument('--device', default='cuda:6')
    parser.add_argument('--input-size', type=int, nargs=2)
    parser.add_argument('--mean', type=float, nargs=3)
    args = parser.parse_args()
    return args


def forward(model, dl, args):
    os.makedirs(args.out_dir, exist_ok=True)

    for sample in tqdm.tqdm(dl, total=len(dl)):
        for i in range(len(sample['id'])):
            id_ = sample['id'][i]
            shape = sample['shape'][i]
            images = sample['image'][i:i+1]
            images = images.to(args.device)
            with torch.no_grad():
                outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            probs = F.interpolate(probs,
                                  size=tuple(shape),
                                  mode='bilinear',
                                  align_corners=True)
            pred = torch.argmax(probs[0], dim=0).cpu().numpy()
            path = os.path.join(args.out_dir, '{}.png'.format(id_))
            cv2.imwrite(path, pred)


def main():
    args = get_args()
    print(args)

    model = torch.load(args.model_path, args.device)
    model.eval()

    dl = DataLoader(Task2Dataset(args.data_dir, 'test',
                                 args.mean, args.input_size))
    forward(model, dl, args)


if __name__ == "__main__":
    main()
