import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import tqdm
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils import data
import sys

def fn_crop(img):
    crop = min(img.shape[0], img.shape[1])
    h, w, = img.shape[0], img.shape[1]
    img = img[(h - crop) // 2:(h + crop) // 2,
        (w - crop) // 2:(w + crop) // 2]
    return img


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--transform', type=str, help='path to images', default='center')
parser.add_argument('--path1', type=str, help='path to images')
parser.add_argument('--path2', type=str, help='path to images')
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')

from cleanfid import fid

if __name__ == '__main__':
    args = parser.parse_args()

    if args.transform == 'center':
        custom_image_tranform = fn_crop
    else:
        custom_image_tranform = None

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args.path1)
    print(args.path2)
    score = fid.compute_fid(args.path1, args.path2, custom_image_tranform=custom_image_tranform)
    print(args.transform)
    print('FID: ', score)