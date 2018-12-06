from os import listdir, makedirs
from os.path import isfile, join, splitext, exists
import argparse
import matplotlib.pyplot as plt
import ast

import numpy as np
from scipy import misc

from preprocess import remove_if_exist

edge = (33 - 21) // 2

def add_edge(sr,scaled):
    scaled[edge:-edge,edge:-edge,:]=sr
    return(scaled)

def add_edge_to_all():
    LR_dir=option.LR_dir
    SR_dir=option.SR_dir+"_nopad"

    for fi in listdir(LR_dir):
        f = join(LR_dir, fi)
        if not isfile(f):
            continue
        X = misc.imread(f, mode='YCbCr')

        srext='.png'
        sr_file = splitext(fi)[0]+srext
        sr_file = join(SR_dir, sr_file)
        remove_if_exist(sr_file)

        # w, h, c = X.shape

        scaled = misc.imresize(X, option.scale/1.0, 'bicubic')
        newshape=list(scaled.shape)
        newimg = np.zeros(newshape)

        misc.imsave(sr_file, newimg)
        # misc.imsave(fi,ycbcr2rgb(scaled))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--padding',
                        default=False,
                        dest='pad',
                        type=ast.literal_eval,
                        help="does the model already padding 0")

    # For multi-file (dir)
    parser.add_argument('-LR', '--LR_dir',
                        default='./data/test_set/LR',
                        dest='LR_dir',
                        type=str,
                        help="LR_dir")
    parser.add_argument('-SR', '--SR_dir',
                        default='./data/test_set/SR',
                        dest='SR_dir',
                        type=str,
                        help="SR_dir")

    option = parser.parse_args()
    if(option.pad):
        edge=0
    # single file
    # predict()
    add_edge_to_all()
