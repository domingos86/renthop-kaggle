import threading

import loaders as l

from . import shuffle, subset

class Generator(object):
    
    def __init__(self, preprocessor, transformations, final_transform = None):
        self.preprocessor = preprocessor
        self.transformations = transformations
        self.final_transform = final_transform
        self.data = None
        self.state = {'prepared': False}
    
    def prepare(self, test = False, batch_size = 128, valid_size = None):
        self.data = self.preprocessor.load_and_transform(test)
        if not test:
            self.data = shuffle(self.data)
        self.state['batch_size'] = batch_size
        if isinstance(self.data, dict):
            self.state['size'] = len(self.data.values()[0])
        else:
            # Here we're assuming it's not a list of features nor a tuple
            self.state['size'] = len(self.data)
        if not test and valid_size is not None:
            self.state['validation_split'] = max(0, self.state['size'] - valid_size)
        else:
            self.state['validation_split'] = self.state['size']
        self.state['prepared'] = True
        
    def get_iterator(self):
        if self.state['prepared']:
            return SubIterator(self, 0, self.state['validation_split'],
                               self.state['batch_size'], self.final_transform)
    
    def get_valid_iterator(self):
        if self.state['prepared']:
            return SubIterator(self, self.state['validation_split'], self.state['size'],
                               self.state['batch_size'], self.final_transform)

    def get_raw_data(self):
        if self.state['prepared']:
            return self.data
        
    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['data']
        del odict['state']
        return odict;

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self.data = None
        self.state = {'prepared': False}

class SubIterator(object):

    def __init__(self, generator, start, end, batch_size, final_transform = None):
        self.generator = generator
        self.start = start
        self.end = end
        self.position = start
        self.batch_size = batch_size
        self.final_transform = final_transform # used for instance to split predictors and response
        self.lock = threading.Lock()

    def n_samples(self):
        return self.end - self.start

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            start = self.position
            self.position += self.batch_size
            if self.position >= self.end:
                self.position = self.start
        end = min(self.end, start + self.batch_size)
        data = subset(self.generator.data, slicer = slice(start, end))
        tr = self.generator.transformations
        for key in tr:
            if isinstance(tr[key], list):
                for transform in tr[key]:
                    data[key] = transform(data[key])
            else:
                data[key] = tr[key](data[key])
        if self.final_transform is not None:
            return self.final_transform(data)
        else:
            return data
