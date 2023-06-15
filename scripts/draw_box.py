import os
import numpy as np
import cv2
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, help='', default='')
parser.add_argument('--insubfolder', type=str, help='', default='top1')
parser.add_argument('--outsubfolder', type=str, help='', default='bbox')
import sys

args = parser.parse_args()

folderlist = args.path
os.system('mkdir -p %s'%(folderlist.replace(args.insubfolder,args.outsubfolder)))
files = [f for f in os.listdir(folderlist) if ('jpg' in f or 'png' in f)]

for imfile in tqdm(files):
    text = imfile.split('_')[0]
    caption = text.split(' <') if ' <' in text else text
    imgs = cv2.imread(os.path.join(folderlist, imfile))
    if imgs is None:
        continue
    w,h = imgs.shape[1], imgs.shape[0]

    box_token = [x for x in text.split(' ') if '<' in x]
    box_token = [float(x[1:-1]) for x in box_token if ('>' in x and x[1:-1].isnumeric())]
    box_list = np.array(box_token[:len(box_token)//4*4]).reshape(-1,4)
    for ii in range(box_list.shape[0]):
        bbox = box_list[ii,:]
        cv2.rectangle(imgs, (int(bbox[0]*w/1000), int(bbox[1]*h/1000)), \
            (int(bbox[2]*w/1000), int(bbox[3]*h/1000)), (11,134,184), int(8*max(h,w)/500))
    cv2.imwrite(os.path.join(folderlist.replace(args.insubfolder,args.outsubfolder), imfile),imgs)