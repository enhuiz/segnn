import os
import sys
import argparse

import torch
import torch.nn as nn

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

from segnn.models.encoder import resnet18
from segnn.models.decoder import PPM, UPerNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', default=os.path.join(root, 'zoo'))
    args = parser.parse_args()
    return args


def autosave(f):
    if not hasattr(autosave, 'funcs'):
        autosave.funcs = []

    def wrapped(out_dir, *args):
        model = f(*args)
        path = os.path.join(out_dir, '{}.pth'.format(f.__name__))
        if not os.path.exists(path):
            print('Making model {}.'.format(path))
            torch.save(model, path)

    autosave.funcs.append(wrapped)

    return wrapped


@autosave
def resnet18_ppm():
    encoder = resnet18(True)
    decoder = PPM(fc_dim=512)
    model = nn.Sequential(encoder, decoder)
    return model


@autosave
def dilated_resnet18_ppm():
    encoder = resnet18(True, dilation=2)
    decoder = PPM(fc_dim=512)
    model = nn.Sequential(encoder, decoder)
    return model


@autosave
def dilated_resnet18_upernet():
    encoder = resnet18(True, dilation=2)
    decoder = UPerNet(fc_dim=512, fpn_inplanes=(64, 128, 256, 512))
    model = nn.Sequential(encoder, decoder)
    return model


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # models are created here
    for func in autosave.funcs:
        func(args.out_dir)


if __name__ == "__main__":
    main()
