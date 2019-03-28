import os
import sys
import argparse

import torch
import torch.nn as nn

from torchvision.models import resnet50, densenet121, resnet18

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from segnn.models.zero_nn import ZeroNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    args = parser.parse_args()
    return args


def autosave(f):
    def wrapper(out_dir, *args):
        model = f(*args)
        path = os.path.join(out_dir, '{}.pth'.format(f.__name__))
        if os.path.exists(path):
            print('{} exists, skip making model.'.format(path))
        else:
            torch.save(model, path)
    return wrapper


@autosave
def zero_nn():
    """
    A model only outputs zero, for debug purpose.
    """
    model = ZeroNN()
    return model


def main():
    args = get_args()

    zero_nn(args.out_dir)


if __name__ == "__main__":
    main()
