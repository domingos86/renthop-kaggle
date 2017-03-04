#!/usr/bin/env python

from sklearn.utils import shuffle as skshuffle

def shuffle(X, y):
	if isinstance(X, dict):
        to_shuffle = X.keys()
        to_shuffle.append(y)
        shuffled = skshuffle(*to_shuffle)
        return dict(zip(X.keys(), shuffled[:-1])), shuffled[-1]
    else:
        return skshuffle(X, y)
