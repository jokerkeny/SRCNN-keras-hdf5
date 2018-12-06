import sys
import os
import h5py
import network
import numpy as np
from scipy import misc

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam

import argparse


def train():

    if(args.pad):
        model = network.srcnn(pad='same')
    else:
        model = network.srcnn()

    output_file = './data.h5'
    h5f = h5py.File(output_file, 'r')
    X = h5f['input']
    y = h5f['label']

    n_epoch = args.n_epoch

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    for epoch in range(0, n_epoch, 5):
        model.fit(X, y, batch_size=128, nb_epoch=5, shuffle='batch')
        if args.save:
            print("Saving model ", epoch + 5)
            model.save(os.path.join(args.save, 'model_%d.h5' %(epoch+5)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--save',
                        default='./save',
                        dest='save',
                        type=str,
                        help="Path to save the checkpoints to")
    parser.add_argument('-D', '--data',
                        default='./dataset/Train/output/',
                        dest='data',
                        type=str,
                        help="Training data directory")
    parser.add_argument('-E', '--epoch',
                        default=50,
                        dest='n_epoch',
                        type=int,
                        help="Training epochs must be a multiple of 5")
    parser.add_argument('-P', '--padding',
                        default=True,
                        dest='pad',
                        type=bool,
                        help="does the model padding 0")

    args = parser.parse_args()
    print(args)
    train()
