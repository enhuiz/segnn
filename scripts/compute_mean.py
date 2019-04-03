import os
import sys
import argparse
from multiprocessing import Pool
import numpy as np
import cv2
import tqdm


root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

from segnn.data import make_samples


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('out_path')
    parser.add_argument('--nj', type=int, default=8)
    args = parser.parse_args()
    return args


def compute_rgb_mean(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image.reshape([-1, 3]).mean(axis=0)


def main():
    args = get_args()
    samples = make_samples(args.data_dir)
    paths = [s[0] for s in samples]
    with Pool(args.nj) as pool:
        mean_rgbs = pool.imap_unordered(compute_rgb_mean, paths)
        mean_rgbs = tqdm.tqdm(mean_rgbs, total=len(paths))
        mean_rgbs = [*mean_rgbs]
    mean_rgb = np.mean(mean_rgbs, axis=0)
    with open(args.out_path, 'w') as f:
        f.write(str(mean_rgb))
        f.write('\n')
    print(mean_rgb)
    print('Result is written to {}.'.format(args.out_path))


if __name__ == "__main__":
    main()
