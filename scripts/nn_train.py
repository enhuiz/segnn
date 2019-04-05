import os
import json
import numpy as np
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from segnn.data import Task2Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--model-path')
    parser.add_argument('--out-dir')
    parser.add_argument('--device', default='cuda:6')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--init-lr', type=float, default=5e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--mean', type=float, nargs=3)
    args = parser.parse_args()
    return args


def train(model, criterion, optimizer, dl, args):
    ckpt_dir = os.path.join(args.out_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for step, sample in enumerate(dl):
            images = sample['image']
            labels = sample['label']
            images = images.to(args.device)
            labels = labels.long().to(args.device)

            outputs = model(images)
            outputs = F.interpolate(outputs,
                                    size=labels.shape[1:],  # i.e. tensor size
                                    mode='bilinear',
                                    align_corners=False)
            # expect the model output logp
            loss = criterion(outputs, labels)
            loss.backward()

            if (step + 1) % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {}'
                  .format(epoch + 1, args.epochs,
                          step + 1, len(dl), loss.item()))

        ckpt_path = os.path.join(ckpt_dir, '{}.pth'.format(epoch + 1))
        torch.save(model, ckpt_path)


def dump_config(args):
    config_path = os.path.join(args.out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f)


def main():
    args = get_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dump_config(args)
    print(args)

    model = torch.load(args.model_path, args.device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    dl = DataLoader(Task2Dataset(args.data_dir, 'train', args.mean))

    train(model, criterion, optimizer, dl, args)


if __name__ == "__main__":
    main()
