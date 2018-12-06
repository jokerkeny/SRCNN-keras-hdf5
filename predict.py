from os import listdir
from os.path import isfile, join
import argparse
import h5py
from keras.models import load_model
import matplotlib.pyplot as plt

import numpy as np
from scipy import misc
import network

input_size = 33
label_size = 21
pad = (33 - 21) // 2

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return rgb.dot(xform.T)


def predict():
    model = network.srcnn((None, None, 1))
    model.load_weights(option.model)

    X = misc.imread(option.input, mode='YCbCr')

    w, h, c = X.shape
    w -= int(w % option.scale)
    h -= int(h % option.scale)
    X = X[0:w, 0:h, :]
    X[:,:,1] = X[:,:,1]
    X[:,:,2] = X[:,:,2]

    scaled = misc.imresize(X, 1.0/option.scale, 'bicubic')
    scaled = misc.imresize(scaled, option.scale/1.0, 'bicubic')
    newshape=list(scaled.shape)
    newshape[0]-=2*pad
    newshape[1]-=2*pad
    newimg = np.zeros(newshape)

    if option.baseline:
        misc.imsave(option.baseline, ycbcr2rgb(scaled[pad:-pad,pad:-pad,:]))

    newimg[:, :, 0, None] = model.predict(scaled[None, :, :, 0, None] / 255)
    newimg[:, :, 1, None] = model.predict(scaled[None, :, :, 1, None] / 255)
    newimg[:, :, 2, None] = model.predict(scaled[None, :, :, 2, None] / 255)
    newimg=ycbcr2rgb(newimg*255)
    misc.imsave(option.output, newimg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', '--model',
                        default='./save/model_50.h5',
                        dest='model',
                        type=str,
                        help="The model to be used for prediction")
    parser.add_argument('-I', '--input-file',
                        default='./dataset/Test/Set5/baby_GT.bmp',
                        dest='input',
                        type=str,
                        help="Input image file path")
    parser.add_argument('-O', '--output-file',
                        default='./dataset/Test/Set5/baby_SRCNN.bmp',
                        dest='output',
                        type=str,
                        help="Output image file path")
    parser.add_argument('-B', '--baseline',
                        default='./dataset/Test/Set5/baby_bicubic.bmp',
                        dest='baseline',
                        type=str,
                        help="Baseline bicubic interpolated image file path")
    parser.add_argument('-S', '--scale-factor',
                        default=3.0,
                        dest='scale',
                        type=float,
                        help="Scale factor")
    option = parser.parse_args()
    predict()
