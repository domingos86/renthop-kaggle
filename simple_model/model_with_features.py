#!/usr/bin/env python

'''    
    If using GPU, should open pyhton interperter with
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python
'''

import os
import sys

import numpy as np

from sklearn.utils import shuffle

from keras.layers.core import Dense, Dropout
from keras.optimizers import Adadelta
from keras.models import Sequential, load_model
from keras import backend as K

import theano

def fit(X, y, plot=False, epochs=3000, save_to=None):
    '''Trains a neural network for all the labels.
    
    It only uses inputs without missing labels.
    
    Returns: trained neural network (nolearn.lasagne.NeuralNet)
    
    Keyword arguments:
    plot -- (bool, False) if true, a plot of the training and validation
        errors at the end of each epoch will be shown once the network
        finishes training.
    epochs -- (int, 3000) the maximum number of epochs for which the
        network should train.
    save_to -- (str, None) name of the file to which the network will be
        pickled.
    '''
    
    X, y = shuffle(X, y)
    
    net = neural_net()

    history = net.fit(X, y, nb_epoch = epochs, batch_size = 128,
                      validation_split = 0.2)

    if save_to:
        net.save(save_to)
    if plot:
        plot_net(history)
    return net, history

def neural_net(initial_rate=0.04):
    net = Sequential()
    net.add(Dense(800, input_shape = (80,), activation = 'relu'))
    net.add(Dropout(0.2))
    net.add(Dense(1000, activation = 'relu'))
    net.add(Dropout(0.5))
    net.add(Dense(200, activation = 'relu'))
    net.add(Dense(3, activation = 'softmax'))
    adadelta = Adadelta(lr = initial_rate)
    net.compile(optimizer = adadelta, loss = 'categorical_crossentropy')
    return net

def predict(net, transform, save_to='submission.csv'):
    X, ids = load(test = True, cols = cols)
    X = normalize_X(X, transform)
    
    y_pred = net.predict(X)

    df = DataFrame(np.hstack(ids, y_pred), columns=['listing_id'] + OUTPUT_COLS)
    df.to_csv(save_to, index=False)
    print("Wrote {}".format(save_to))

def plot_net(history):
    from matplotlib import pyplot
    train_loss = history.history.items()[0][1]
    valid_loss = history.history.items()[1][1]
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(4e-1, 1)
    pyplot.yscale("log")
    try:
        pyplot.show()
    except RuntimeError as e:
        print "Unable to show plot", e

def save_net(net, file_name):
    net.save(file_name)

def load_net(file_name):
    return load_model(file_name)

def float32(k):
    return np.cast['float32'](k)
