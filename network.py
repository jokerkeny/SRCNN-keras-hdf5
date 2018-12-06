from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam


def srcnn(input_shape=(33,33,1),pad='valid'):
    model = Sequential()
    model.add(Convolution2D(64, 9, 9, border_mode=pad, input_shape=input_shape, activation='relu'))
    model.add(Convolution2D(32, 1, 1, border_mode=pad, activation='relu'))
    model.add(Convolution2D(1, 5, 5, border_mode=pad))
    model.compile(Adam(lr=0.001), 'mse')
    return model
