#!/usr/bin/env python

'''    
    If using GPU, should open pyhton interperter with
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python
'''

import os
import sys

import numpy as np
from pandas import DataFrame

from ..preprocessing import shuffle

from keras.layers import Dense, Dropout, Embedding, InputLayer, Merge, Flatten
from keras.optimizers import Adadelta
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import backend as K

import theano

OUTPUT_COLS = ['high', 'medium', 'low']

def fit(X, y, plot=False, epochs=3000, save_to='nn_trained', final = False):
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
    
    net = neural_net2()
    
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    save_to = save_to + '/'
    
    earlystopping = EarlyStopping(patience = 20)
    checkpoint = ModelCheckpoint(save_to + 'net.h5', save_best_only = True)
    logger = CSVLogger(save_to + 'history.log')
    lrreducer = ReduceLROnPlateau(factor = 0.2, patience = 10)
    callbacks = [checkpoint, logger, earlystopping, lrreducer]
    
    history = net.fit(X, y, nb_epoch = epochs, batch_size = 128,
                      validation_split = 0 if final else 0.1,
                      callbacks = callbacks)

    if plot:
        plot_net(history)
    return net, history

def neural_net1(initial_rate=0.04):
    embedding = Sequential()
    embedding.add(InputLayer((1,), name = 'managers'))
    embedding.add(Embedding(1000, 10, input_length = 1))
    embedding.add(Flatten())
    embedding.add(Dropout(0.3))
    main_input = Sequential()
    main_input.add(InputLayer((80,), name = 'main'))
    net = Sequential()
    net.add(Merge([main_input, embedding], mode = 'concat'))
    net.add(Dense(800, activation = 'relu'))
    net.add(Dropout(0.2))
    net.add(Dense(1000, activation = 'relu'))
    net.add(Dropout(0.5))
    net.add(Dense(200, activation = 'relu'))
    net.add(Dense(3, activation = 'softmax'))
    adadelta = Adadelta(lr = initial_rate)
    net.compile(optimizer = adadelta, loss = 'categorical_crossentropy')
    return net

def neural_net2(initial_rate=0.04):
    embedding = Sequential()
    embedding.add(InputLayer((1,), name = 'managers'))
    embedding.add(Embedding(1000, 10, input_length = 1))
    embedding.add(Flatten())
    embedding.add(Dropout(0.3))
    main_input = Sequential()
    main_input.add(InputLayer((86,), name = 'main'))
    net = Sequential()
    net.add(Merge([main_input, embedding], mode = 'concat'))
    net.add(Dense(800, activation = 'relu'))
    net.add(Dropout(0.2))
    net.add(Dense(1000, activation = 'relu'))
    net.add(Dropout(0.5))
    net.add(Dense(200, activation = 'relu'))
    net.add(Dense(3, activation = 'softmax'))
    adadelta = Adadelta(lr = initial_rate)
    net.compile(optimizer = adadelta, loss = 'categorical_crossentropy')
    return net

def predict(net, X, ids, save_to='submission.csv'):
    y_pred = net.predict(X)

    df = DataFrame(np.hstack([ids.reshape(-1, 1), y_pred]), columns=['listing_id'] + OUTPUT_COLS)
    df['listing_id'] = df['listing_id'].astype('int')
    df.to_csv(save_to, index=False)
    print("Wrote {}".format(save_to))

def plot_net(history, save_to = None):
    from matplotlib import pyplot
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(4e-1, 1)
    pyplot.yscale("log")
    try:
        if save_to:
            pyplot.savefig(save_to)
        else:
            pyplot.show()
    except RuntimeError as e:
        print "Unable to show plot", e

def save_net(net, file_name):
    net.save(file_name)

def load_net(file_name):
    return load_model(file_name)
