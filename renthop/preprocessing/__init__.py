#!/usr/bin/env python

from sklearn.utils import shuffle as skshuffle

def shuffle(X, y = None):
    if isinstance(X, dict):
        to_shuffle = X.values()
        if y is not None:
            to_shuffle.append(y)
        shuffled = skshuffle(*to_shuffle)
        if y is not None:
            return dict(zip(X.keys(), shuffled[:-1])), shuffled[-1]
        else:
            return dict(zip(X.keys(), shuffled))
    else:
        return skshuffle(X, y)

def subset(X, y = None, slicer = slice(None, None)):
    if isinstance(X, dict):
        sliced = X.copy()
        for key in sliced:
            sliced[key] = subset(sliced[key], slicer = slicer)
        if y is not None:
            return sliced, subset(y, slicer = slicer)
        else:
            return sliced
    else:
        if y is not None:
            return subset(X, slicer = slicer), subset(y, slicer = slicer)
        else:
            return X[slicer]
