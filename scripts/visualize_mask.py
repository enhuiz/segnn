import cv2
import argparse

import os
import sys
import argparse

import torch
import torch.nn as nn
import tqdm

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

from segnn.utils import append_mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir')
    parser.add_argument('mask_dir')
    parser.add_argument('out_dir')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    filenames = os.listdir(args.mask_dir)
    for filename in tqdm.tqdm(filenames):
        mask_path = os.path.join(args.mask_dir, filename)
        image_path = os.path.join(args.image_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = append_mask(image, mask)
        out_path = os.path.join(args.out_dir, filename)
        cv2.imwrite(out_path, image)

    print('done.')

if __name__ == "__main__":
    main()
