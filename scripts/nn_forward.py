import os
import numpy as np
import argparse
import sys

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from segnn.data import make_test_dl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--model-path')
    parser.add_argument('--out-dir')
    parser.add_argument('--device', default='cuda:6')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--mean', type=float, nargs=3)
    parser.add_argument('--tensor-size', type=int, nargs=2)
    args = parser.parse_args()
    return args


def forward(model, dl, args):
    os.makedirs(args.out_dir, exist_ok=True)

    for step, (ids, sizes, images, _) in enumerate(dl):
        images = images.to(args.device)
        with torch.no_grad():
            outputs = model(images)

        for i in range(len(ids)):
            id_, size, output = ids[i], tuple(sizes[i]), outputs[i:i+1]
            output = F.interpolate(output,
                                   size=size,
                                   mode='bilinear',
                                   align_corners=True)
            output = torch.argmax(output, dim=1).cpu().numpy()[0]
            path = os.path.join(args.out_dir, '{}.png'.format(id_))
            cv2.imwrite(path, output)

        print('Step [{}/{}]'.format(step + 1, len(dl)))


def main():
    args = get_args()
    model = torch.load(args.model_path, args.device)
    model.eval()
    dl = make_test_dl(args.data_dir,
                      args.batch_size,
                      args.mean,
                      args.tensor_size)
    forward(model, dl, args)


if __name__ == "__main__":
    main()
