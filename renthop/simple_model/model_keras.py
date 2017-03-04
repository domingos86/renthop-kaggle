#!/usr/bin/env python

'''    
    If using GPU, should open pyhton interperter with
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python
'''

import os
import sys

import numpy as np
from pandas.io.json import read_json
from pandas import DataFrame, get_dummies

from sklearn.utils import shuffle

from keras.layers.core import Dense, Dropout
from keras.optimizers import Adadelta
from keras.models import Sequential, load_model
from keras import backend as K

import theano

from datetime import datetime
import re

FTRAIN = 'data/train.json'
FTEST = 'data/test.json'

np.random.seed(42)

COLS = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'price_per_bathroom', 'price_per_bedroom', 'day_of_month',
            'hour', 'day_of_week', 'desc_len', 'num_features', 'features_len', 'num_photos']

OUTPUT_COLS = ['high', 'medium', 'low']

def load(test = False, cols = COLS):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_json(os.path.expanduser(fname)) # load pandas dataframe

    df['price_per_bathroom'] = np.log((df['price']+1)/(df['bathrooms']+1))
    df['price_per_bedroom'] = np.log((df['price']+1)/(df['bedrooms']+1))
    df['price'] = np.log(df['price'] + 1)
    df['created'] = df['created'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['day_of_month'] = df['created'].apply(lambda x: x.day)
    df['hour'] = df['created'].apply(lambda x: x.hour + (x.minute + x.second / 60.0) / 60.0)
    df['day_of_week'] = df['created'].apply(lambda x: x.weekday())
    df['desc_len'] = df['description'].apply(lambda desc: len([x for x in re.split(r'\W+', desc) if len(x) > 0]))
    df['num_features'] = df['features'].apply(len)
    df['features_len'] = df['features'].apply(lambda feats: sum([len([x for x in re.split(r'\W+', feat) if len(x) > 0]) for feat in feats]))
    df['num_photos'] = df['photos'].apply(len)
    # force all coordinates within NYC area
    df['longitude'] = df['longitude'].apply(bound(-74.3434, -73.62))
    df['latitude'] = df['latitude'].apply(bound(40.4317, 41.0721))

    print(df.count())  # prints the number of values for each column
    
    if not test:  # only FTRAIN has any target columns
        df = df.dropna()  # drop all rows that have missing values in them
        X = np.array(df[cols], dtype = np.float32)
        y = np.array(get_dummies(df['interest_level'])[OUTPUT_COLS], np.float32)
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
    else:
        X = np.array(df[cols], dtype = np.float32)
        y = df['listing_id'].as_matrix()

    return X, y

def normalize_X(X, transform = None):
    if not transform:
        transform = (X.min(axis = 0), X.max(axis = 0))
    X = (X - transform[0]) / (transform[1] - transform[0])
    return X, transform

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
    net.add(Dense(800, input_shape = (14,), activation = 'relu'))
    net.add(Dropout(0.2))
    net.add(Dense(100, activation = 'relu'))
    net.add(Dropout(0.2))
    net.add(Dense(200, activation = 'relu'))
    net.add(Dense(3, activation = 'softmax'))
    adadelta = Adadelta(lr = initial_rate)
    net.compile(optimizer = adadelta, loss = 'categorical_crossentropy')
    return net

def predict(net, transform, save_to='submission.csv', cols = COLS):
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

def bound(m, M):
    return lambda x: max(min(x, M), m)
