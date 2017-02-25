#!/usr/bin/env python

'''
    Based mostly on the code found in https://github.com/dnouri/kfkd-tutorial
    
    To run, open python interpreter, import location_predictor and run
    location_predictor.fit().
    
    If using GPU, should open pyhton interperter with
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python
'''

import os
import sys

import numpy as np
from pandas.io.json import read_json
from pandas import DataFrame, get_dummies
from sklearn.utils import shuffle
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy
from nolearn.lasagne import NeuralNet, TrainSplit
import theano
import cPickle as pickle

from datetime import datetime
import re

FTRAIN = 'data/train.json'
FTEST = 'data/test.json'
# FLOOKUP = 'IdLookupTable.csv'

sys.setrecursionlimit(10000)  # for pickle...
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

    df['price_per_bathroom'] = df['price']/(df['bathrooms']+1)
    df['price_per_bedroom'] = df['price']/(df['bedrooms']+1)
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
    net = neural_net(epochs)

    net.fit(X, y)

    if save_to:
        save_net(net, save_to)
    if plot:
        plot_net(net)
    return net

def neural_net(epochs=3000, initial_rate=0.04):
    return NeuralNet(
        layers = [
            ('input', layers.InputLayer),
            ('hidden1', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('hidden2', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('hidden3', layers.DenseLayer),
            ('output', layers.DenseLayer)],
        input_shape=(None, 14),
        hidden1_num_units = 800,
        dropout1_p = 0.2,
        hidden2_num_units = 1000,
        dropout2_p = 0.2,
        hidden3_num_units = 200,
        output_num_units = 3, output_nonlinearity = softmax,
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(initial_rate)),
        update_momentum=theano.shared(float32(0.9)),
        max_epochs = epochs,
        verbose = 1,
        regression = True,
        on_epoch_finished=[
                LinearAdjustVariable('update_learning_rate', start=initial_rate, stop=0.0001),
                LinearAdjustVariable('update_momentum', start=0.9, stop=0.999),
                EarlyStopping(patience=100)],
        objective_loss_function = categorical_crossentropy)

def predict(net, transform, save_to='submission.csv', cols = COLS):
    X, ids = load(test = True, cols = cols)
    X = normalize_X(X, transform)
    
    y_pred = net.predict(X)

    df = DataFrame(np.hstack(ids, y_pred), columns=['listing_id'] + OUTPUT_COLS)
    df.to_csv(save_to, index=False)
    print("Wrote {}".format(save_to))

class LinearAdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def plot_net(net):
    from matplotlib import pyplot
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
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
    with open(file_name, 'wb') as f:
        pickle.dump(net, f, -1)

def load_net(file_name):
    with open(file_name, 'rb') as f:
        net = pickle.load(f)
    return net

def float32(k):
    return np.cast['float32'](k)

def bound(m, M):
    return lambda x: max(min(x, M), m)
