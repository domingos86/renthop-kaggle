#!/usr/bin/env python

'''    
    If using GPU, should open pyhton interperter with
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python
'''

import os
import sys

import numpy as np
from pandas import DataFrame

from ..preprocessing import shuffle, loaders

from keras.layers import Dense, Dropout, Embedding, InputLayer, Merge, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adadelta
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import backend as K

import theano

OUTPUT_COLS = ['high', 'medium', 'low']

def fit(X, y, plot=False, epochs=3000, save_to='nn_trained'):
    '''Trains a neural network for all the labels.
        
    Keyword arguments:
    plot -- (bool, False) if true, a plot of the training and validation
        errors at the end of each epoch will be shown once the network
        finishes training.
    epochs -- (int, 3000) the maximum number of epochs for which the
        network should train.
    save_to -- (str, 'nn_trained') name of the directory in which the
        training history and network are saved.
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
                      validation_split = 0.1,
                      callbacks = callbacks)

    if plot:
        plot_net(history)
    return net, history
    
def fit_generator(train_generator, valid_generator = None, plot=False,
                        epochs=3000, save_to='nn_trained',
                        lr_reduce_after = 20, early_stopping = 50):
    '''Trains a neural network for all the labels.
        
    Keyword arguments:
    plot -- (bool, False) if true, a plot of the training and validation
        errors at the end of each epoch will be shown once the network
        finishes training.
    epochs -- (int, 3000) the maximum number of epochs for which the
        network should train.
    save_to -- (str, 'nn_trained') name of the directory in which the
        training history and network are saved.
    '''
    
    net = neural_net_photo()
    
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    save_to = save_to + '/'
    
    earlystopping = EarlyStopping(patience = early_stopping)
    checkpoint = ModelCheckpoint(save_to + 'net.h5', save_best_only = True)
    logger = CSVLogger(save_to + 'history.log')
    lrreducer = ReduceLROnPlateau(factor = 0.2, patience = lr_reduce_after)
    callbacks = [checkpoint, logger, earlystopping, lrreducer]
    
    history = net.fit_generator(train_generator, train_generator.n_samples(),
            nb_epoch = epochs, callbacks = callbacks, validation_data = valid_generator,
            nb_val_samples = valid_generator.n_samples() if valid_generator is not None else None,
            max_q_size=10, nb_worker=5, pickle_safe = True)

    if plot:
        plot_net(history)
    return net, history

def neural_net1(initial_rate=0.04):
    photo = Sequential()
    photo.add(InputLayer((3, 100, 100), name = 'photo_cover')) #3*100*100
    photo.add(Convolution2D(32, 5, 5, activation = 'relu')) #32*96*96
    photo.add(MaxPooling2D(pool_size = (2, 2))) #32*48*48
    photo.add(Convolution2D(64, 3, 3, activation = 'relu')) #64*46*46
    photo.add(MaxPooling2D(pool_size = (2, 2))) #64*23*23
    photo.add(Convolution2D(32, 4, 4, activation = 'relu')) #32*20*20
    photo.add(MaxPooling2D(pool_size = (2, 2))) #32*10*10
    photo.add(Flatten())
    img_size = Sequential()
    img_size.add(InputLayer((3,), name = 'photo_cover_stats'))
    img_merged = Sequential()
    img_merged.add(Merge([photo, img_size], mode = 'concat')) #3203
    img_merged.add(Dense(20, activation = 'sigmoid')) #20
    img_merged.add(Dropout(0.2))
    embedding = Sequential()
    embedding.add(InputLayer((1,), name = 'managers'))
    embedding.add(Embedding(1000, 10, input_length = 1))
    embedding.add(Flatten())
    embedding.add(Dropout(0.3))
    main_input = Sequential()
    main_input.add(InputLayer((83,), name = 'main'))
    net = Sequential()
    net.add(Merge([main_input, embedding, img_merged], mode = 'concat'))
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
    photo = Sequential()
    photo.add(InputLayer((3, 100, 100), name = 'photo_cover')) #3*100*100
    photo.add(Convolution2D(32, 5, 5, activation = 'relu')) #32*96*96
    photo.add(MaxPooling2D(pool_size = (2, 2))) #32*48*48
    photo.add(Convolution2D(64, 3, 3, activation = 'relu')) #64*46*46
    photo.add(MaxPooling2D(pool_size = (2, 2))) #64*23*23
    photo.add(Convolution2D(32, 4, 4, activation = 'relu')) #32*20*20
    photo.add(MaxPooling2D(pool_size = (2, 2))) #32*10*10
    photo.add(Flatten())
    img_size = Sequential()
    img_size.add(InputLayer((3,), name = 'photo_cover_stats'))
    img_merged = Sequential()
    img_merged.add(Merge([photo, img_size], mode = 'concat')) #3203
    img_merged.add(Dense(20, activation = 'relu')) #20
    img_merged.add(Dropout(0.2))
    embedding = Sequential()
    embedding.add(InputLayer((1,), name = 'managers'))
    embedding.add(Embedding(1000, 10, input_length = 1))
    embedding.add(Flatten())
    embedding.add(Dropout(0.3))
    main_input = Sequential()
    main_input.add(InputLayer((83,), name = 'main'))
    net = Sequential()
    net.add(Merge([main_input, embedding], mode = 'concat'))
    net.add(Dense(780, activation = 'relu'))
    net.add(Dropout(0.2))
    net2 = Sequential()
    net2.add(Merge([net, img_merged], mode = 'concat'))
    net2.add(Dense(1000, activation = 'relu'))
    net2.add(Dropout(0.5))
    net2.add(Dense(200, activation = 'relu'))
    net2.add(Dense(3, activation = 'softmax'))
    adadelta = Adadelta(lr = initial_rate)
    net2.compile(optimizer = adadelta, loss = 'categorical_crossentropy')
    return net2

def neural_net_photo(initial_rate=0.04):
    photo = Sequential()
    photo.add(InputLayer((3, 100, 100), name = 'photo')) #3*100*100
    photo.add(Convolution2D(32, 5, 5, activation = 'relu')) #32*96*96
    photo.add(MaxPooling2D(pool_size = (2, 2))) #32*48*48
    photo.add(Dropout(0.25))
    photo.add(Convolution2D(64, 3, 3, activation = 'relu')) #64*46*46
    photo.add(MaxPooling2D(pool_size = (2, 2))) #64*23*23
    photo.add(Dropout(0.25))
    photo.add(Convolution2D(64, 4, 4, activation = 'relu')) #64*20*20
    photo.add(MaxPooling2D(pool_size = (2, 2))) #64*10*10
    photo.add(Dropout(0.25))
    photo.add(Flatten())
    img_size = Sequential()
    img_size.add(InputLayer((3,), name = 'photo_stats'))
    img_merged = Sequential()
    img_merged.add(Merge([photo, img_size], mode = 'concat')) #6403
    img_merged.add(Dense(500, activation = 'relu')) #500
    img_merged.add(Dropout(0.5))
    img_merged.add(Dense(20, activation = 'relu')) #20
    img_merged.add(Dense(3, activation = 'softmax'))
    adadelta = Adadelta(lr = initial_rate)
    img_merged.compile(optimizer = adadelta, loss = 'categorical_crossentropy')
    return img_merged

def predict(net, X, ids, photo_data, save_to='submission.csv', load_batch_size = 1000):
    photo_loader = loaders.PhotoLoader()
    y_pred = np.zeros((0, 3), dtype = float32)
    for i in range(0, ids.shape[0], load_batch_size):
        X_ = _slice_dict(X, slice(i,(i + load_batch_size)))
        photo_loader.consume(X_['photo_cover'], None)
        X_['photo_cover'] = photo_loader()
        y_pred = np.vstack((y_pred, net.predict(X_)))

    df = DataFrame(np.hstack([ids.reshape(-1, 1), y_pred]), columns=['listing_id'] + OUTPUT_COLS)
    df['listing_id'] = df['listing_id'].astype('int')
    df.to_csv(save_to, index=False)
    print("Wrote {}".format(save_to))

def predict_generator(net, generator, ids, save_to='submission.csv'):
    y_pred = net.predict_generator(generator, val_samples = generator.n_samples(),
                            max_q_size = 10, nb_worker=5, pickle_safe = True)
    df = DataFrame(np.hstack([ids.reshape(-1, 1), y_pred]), columns=['listing_id'] + OUTPUT_COLS)
    df['listing_id'] = df['listing_id'].astype('int')
    df.to_csv(save_to, index=False)
    print("Wrote {}".format(save_to))

def _slice_dict(idict, sl):
    odict = {}
    for key in idict:
        odict[key] = idict[key][sl]
    return odict

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
