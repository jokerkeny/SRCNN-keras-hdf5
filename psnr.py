import math
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists, splitext
import os
import argparse
import cv2

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_dir(option):
    HR_dir=option.HR_dir
    SR_dir=option.SR_dir
    psnr_list=[]

    srext=splitext(listdir(SR_dir)[0])[1]

    for fi in listdir(HR_dir):
        f = join(HR_dir, fi)
        if not isfile(f):
            continue
        hrimage = cv2.imread(f)

        f = splitext(fi)[0]+srext
        f = join(SR_dir, f)
        if not isfile(f):
            continue
        srimage = cv2.imread(f)
        psnr_list.append(psnr(hrimage,srimage))

    return np.mean(psnr_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-HR', '--HR_dir',
                        default='./data/test_set/HR/',
                        dest='HR_dir',
                        type=str,
                        help="HR_dir")
    parser.add_argument('-SR', '--SR_dir',
                        default='./data/test_set/SR/',
                        dest='SR_dir',
                        type=str,
                        help="SR_dir")
    option = parser.parse_args()
    print(psnr_dir(option))