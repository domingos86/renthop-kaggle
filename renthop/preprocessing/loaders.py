import features

import numpy as np
from pandas.io.json import read_json
from pandas import DataFrame, get_dummies, read_csv

from sklearn.utils import shuffle, resample

from datetime import datetime
import re, os

from PIL import Image

try:
    import cPickle as pickle
except:
    import pickle

OUTPUT_COLS = ['high', 'medium', 'low']

class Preprocessor(object):

    def __init__(self):
        self._fitted = False
        
        self._pipelines = []
        self._base_pipeline_operations = []
        self._base_pipeline_loader = None
        self._base_pipeline_only_train = False
        self._cur_pipeline = None

    def with_pipeline(self, pipeline = None):
        self._cur_pipeline = pipeline
        return self

    def set_loader(self, loader, only_train = False, pipeline = None):
        if pipeline is None:
            pipeline = self._cur_pipeline
        if self._fitted:
            raise Exception('Cannot set loader to an already fitted preprocessor!')
        if not callable(loader):
            raise Exception('Loader must be callable!')
        if pipeline is None:
            self._base_pipeline_loader = loader
            self._base_pipeline_only_train = only_train
        if pipeline is None and len(self._pipelines) == 0:
            pipeline = 'main'
        if pipeline is not None:
            self._create_pipeline_if_necessary(pipeline)
            pipeline = [pipeline]
        else:
            pipeline = self._pipelines
        for p in pipeline:
            setattr(self, 'loader_' + p, loader)
            setattr(self, 'only_train_' + p, only_train)
        return self

    def add_operation(self, operation, pipeline = None):
        if pipeline is None:
            pipeline = self._cur_pipeline
        if self._fitted:
            raise Exception('Cannot add operations to an already fitted preprocessor!')
        methods = dir(operation)
        if 'fit' not in methods or 'transform' not in methods:
            raise Exception('operation must implement fit and transform methods!')
        if pipeline is None:
            self._base_pipeline_operations.append(operation)
        if pipeline is None and len(self._pipelines) == 0:
            pipeline = 'main'
        if pipeline is not None:
            if pipeline not in self._pipelines:
                self._create_pipeline_if_necessary(pipeline)
                return
            pipeline = [pipeline]
        else:
            pipeline = self._pipelines
        for p in pipeline:
            if not hasattr(self, 'operations_' + p):
                setattr(self, 'operations_' + p, [])
            getattr(self, 'operations_' + p).append(operation)
        return self

    def set_consumer(self, consumer, pipeline = None):
        if pipeline is None:
            pipeline = self._cur_pipeline
        if self._fitted:
            raise Exception('Cannot set consumer to an already fitted preprocessor!')
        if 'consume' not in dir(consumer):
            raise Exception('Consumer must implement consume method!')
        if pipeline is None:
            self._base_pipeline_operations.append(operation)
        if pipeline is None and (len(self._pipelines) or\
                len(self._pipelines) == 1 and self._pipelines[0] == 'main') == 0:
            pipeline = 'main'
        if pipeline is not None:
            if pipeline not in self._pipelines:
                self._create_pipeline_if_necessary(pipeline)
            pipeline = [pipeline]
        else:
            raise Exception('Cannot set consumer to all pipelines!')
        for p in pipeline:
            setattr(self, 'consumer_' + p, consumer)
        return self

    def load_and_transform(self, test = False, verbose = 0):
        if test and not self._fitted:
            raise Exception('Cannot transform a test set before the transforms are fitted!')
        if not self._pipelines:
            raise Exception('No loaders were set!')
        for pipeline in self._pipelines:
            if not hasattr(self, 'loader_' + pipeline):
                raise Exception('Pipeline ' + pipeline + ' has no loader!')
        loaded_data = {}
        for pipeline in self._pipelines:
            if test and getattr(self, 'only_train_' + pipeline):
                if verbose == 1:
                    print "Pipeline", pipeline, "ignored as it's only set for train data."
                continue
            if verbose == 1:
                print "Pipeline", pipeline, "started."
            data = getattr(self, 'loader_' + pipeline)(test)
            if verbose == 1:
                print "Pipeline", pipeline, "loaded."
            if hasattr(self, 'operations_' + pipeline):
                for operation in getattr(self, 'operations_' + pipeline):
                    if not test:
                        operation.fit(data)
                    data = operation.transform(data)
                    if verbose == 1:
                        print "Pipeline", pipeline + ':', "done", str(operation)
            if hasattr(self, 'consumer_' + pipeline):
                data = getattr(self, 'consumer_' + pipeline).consume(data, pipeline)
            if data is not None:
                loaded_data[pipeline] = data
            if verbose == 1:
                print "Pipeline", pipeline, "finished."
        if len(loaded_data) == 1:
            loaded_data = loaded_data[loaded_data.keys()[0]]
        self._fitted = True
        return loaded_data

    def _create_pipeline_if_necessary(self, pipeline):
        if pipeline not in self._pipelines:
            self._pipelines.append(pipeline)
            setattr(self, 'loader_' + pipeline, self._base_pipeline_loader)
            setattr(self, 'operations_' + pipeline,
                    [operator for operator in self._base_pipeline_operations])
            setattr(self, 'only_train_' + pipeline, self._base_pipeline_only_train)

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        result = None
        with open(file_name, 'rb') as f:
            result = pickle.load(f)
        return result

class BaseLoader(object):

    def __init__(self, ftrain, ftest, sample = None):
        self.ftrain = ftrain
        self.ftest = ftest
        self.sample = sample
        self.clear()

    def __call__(self, test = False):
        if test:
            if self.test is None:
                self.test = self._load_dataframe(self.ftest)
            return self.test
        else:
            if self.train is None:
                self.train = self._load_dataframe(self.ftrain)
                if self.sample:
                    self.train = self.train.sample(self.sample)
            return self.train

    def select_loader(self, columns):
        return SelectorLoader(self, columns)

    @staticmethod
    def _load_dataframe(fname):
        raise NotImplementedError

    def clear(self):
        self.train = None
        self.test = None

    def __getstate__(self):
        # Make sure we're not pickling the dataframes
        odict = self.__dict__.copy()
        del odict['train']
        del odict['test']
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self.clear()

class JSONLoader(BaseLoader):

    def __init__(self, ftrain = 'data/train.json', ftest = 'data/test.json',
                 sample = None):
        super(JSONLoader, self).__init__(ftrain, ftest, sample)

    @staticmethod
    def _load_dataframe(fname):
        df = read_json(os.path.expanduser(fname))
        df['created'] = df['created'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        return df

class CSVLoader(BaseLoader):
    @staticmethod
    def _load_dataframe(fname):
        return read_csv(os.path.expanduser(fname))

class SelectorLoader(object):
    # This class replaces a lambda expression, that I'm concerned would
    # break the pickling. Also, a normal function would not suffice
    # in that the state (columns) must be preserved.

    def __init__(self, loader, columns):
        self.loader = loader
        self.columns = columns

    def __call__(self, test = False):
        return self.loader(test)[self.columns]

class Selector(object):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, data):
        pass

    def transform(self, data):
        return data[self.columns]

class ColumnDrop(object):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, data):
        pass

    def transform(self, data):
        return data.drop(self.columns, axis = 1)

class ToNdarray(object):
    def __init__(self, dtype = np.float32, outshape = None):
        self.dtype = dtype
        self.outshape = outshape
    
    def fit(self, data):
        pass

    def transform(self, data):
        data = np.array(data, dtype = self.dtype)
        if self.outshape:
            data = data.reshape(self.outshape)
        return data

class Slicer(object):
    def __init__(self, rowslice, colslice):
        self.rowslice = rowslice
        self.colslice = colslice
    
    def fit(self):
        pass
    
    def transform(self, data):
        return data[rowslice, colslice]

class BasePipelineMerger(object):
    '''
    Subclasses must implement do_merge(self, test = False) that would set
    how to combine data from self.data.
    '''
    def __init__(self, input_pipelines):
        self.input_pipelines = input_pipelines
        self.data = {}

    def consume(self, data, pipeline):
        self.data[pipeline] = data

    def do_merge(self, test = False):
        raise NotImplementedError

    def __call__(self, test = False):
        return self.do_merge(test)

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['data']
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)
        self.data = {}

class PandasColumnMerger(BasePipelineMerger):

    def __init__(self, input_pipelines, on = None, how = 'outer'):
        super(PandasColumnMerger, self).__init__(input_pipelines)
        self.on = on
        self.how = how

    def do_merge(self, test = False):
        if self.on:
            df = self.data[self.input_pipelines[0]]
            for pipeline in self.input_pipelines[1:]:
                df = df.merge(self.data[pipeline], how = self.how,
                              on = self.on)
            return df
        else:
            return pd.concat(self.data.values(), ignore_index = True)

class GetTopPhotoMerger(BasePipelineMerger):
    def __init__(self, values, urls):
        super(GetTopPhotoMerger, self).__init__([values, urls])
        self.values = values
        self.urls = urls
        self.train_result = None
        self.test_result = None
    
    def do_merge(self, test = False):
        if test:
            if self.test_result is not None:
                return self.test_result
        else:
            if self.train_result is not None:
                return self.train_result
        images = self.data[self.values]
        images['sharpness'] = np.sqrt(images['sharpness'])
        aggregates = images.groupby('listing_id')[['width', 'height', 'sharpness']].aggregate(np.mean).reset_index()
        aggregates = aggregates.rename(columns = {'width': 'avg_width', 'height': 'avg_height', 'sharpness': 'avg_sharpness'})
        urls = self.data[self.urls]
        urls['photo_name'] = urls['photos'].apply(lambda x: x[0] if len(x) > 0 else '').apply(lambda x: re.sub(r'^.*/', '', x))
        first_photos = urls[['listing_id', 'photo_name']].merge(images, how = 'inner', on = ['listing_id', 'photo_name'])
        first_photos = first_photos.rename(columns = {'width': 'cover_width', 'height': 'cover_height', 'sharpness': 'cover_sharpness'})
        result = urls[['listing_id']].merge(aggregates, how = 'left', on = 'listing_id')
        result = result.merge(first_photos[['listing_id', 'photo_name', 'cover_width', 'cover_height', 'cover_sharpness']],
                                how = 'left', on = 'listing_id')
        result['photo_name'] = result['photo_name'].fillna('')
        result = result.fillna(0.0)
        if test:
            self.test_result = result
        else:
            self.train_result = result
        return result
    
    def __getstate__(self):
        odict = super(GetTopPhotoMerger, self).__getstate__()
        del odict['test_result']
        del odict['train_result']
        return odict

    def __setstate__(self, idict):
        super(GetTopPhotoMerger, self).__setstate__(idict)
        self.train_result = None
        self.test_result = None

class PhotoLoader(object):
    def consume(self, data, pipeline):
        self.data = data
    
    def __call__(self, test = False):
        if test:
            # Test data doesn't get sampled, so it should be loaded by batches
            return self.data
        else:
            images = np.zeros((self.data.shape[0], 3, 100, 100), dtype = np.float32)
            for i, row in enumerate(self.data.iterrows()):
                images[i] = _get_image(row[1]['listing_id'], row[1]['photo_name'])
            images = images / 255.0
            return images

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['data']
        return odict

    def __setstate__(self, idict):
        self.__dict__.update(idict)

class PhotoLoaderGenerator(object):
    def __call__(self, data):
        images = np.zeros((data.shape[0], 3, 100, 100), dtype = np.float32)
        for i, row in enumerate(data.iterrows()):
            images[i] = _get_image(row[1]['listing_id'], row[1]['photo_name'])
        images = images / 255.0
        return images

def _get_image(listing_id, photo_name):
    if len(photo_name) == 0:
        return np.zeros((3, 100, 100))
    path = 'data/images_compressed/%d/%d/%s' % (listing_id / 1000,
                    listing_id, photo_name)
    im = Image.open(path)
    pixels = np.array(im.getdata()).reshape(10000, -1).swapaxes(0, 1).reshape(-1, 100, 100)
    if pixels.shape[0] == 1:
        pixels = np.repeat(pixels, 3, axis = 0)
    return pixels.astype(np.float32)

class DateTimeExtractor(object):
    operations = {
        'year': lambda x: x.year,
        'month': lambda x: x.month,
        'day_of_month': lambda x: x.day,
        'hour': lambda x: x.hour + (x.minute + x.second / 60.0) / 60.0,
        'day_of_week': lambda x: x.weekday()
    }

    def __init__(self, fields = ['month', 'day_of_month', 'hour', 'day_of_week'],
                    datetime_field = 'created'):
        self.fields = fields
        self.datetime_field = datetime_field

    def fit(self, data):
        pass

    def transform(self, data):
        for field in self.fields:
            data[field] = data[self.datetime_field].apply(self.operations[field])
        return data

class NewSimplePredictors(object):
    # This is an operation
    
    def fit(self, data):
        pass
    
    def transform(self, df):
        df['price_per_bathroom'] = df['price']/(df['bathrooms']+1)
        df['price_per_bedroom'] = df['price']/(df['bedrooms']+1)
        df['desc_len'] = df['description'].apply(lambda desc: len([x for x in re.split(r'\W+', desc) if len(x) > 0]))
        df['num_features'] = df['features'].apply(len)
        df['features_len'] = df['features'].apply(lambda feats: sum([len([x for x in re.split(r'\W+', feat) if len(x) > 0]) for feat in feats]))
        df['num_photos'] = df['photos'].apply(len)
        # force all coordinates within NYC area
        df['longitude'] = df['longitude'].apply(self._bound(-74.3434, -73.62))
        df['latitude'] = df['latitude'].apply(self._bound(40.4317, 41.0721))
        return df
    
    @staticmethod
    def _bound(m, M):
        return lambda x: max(min(x, M), m)

class Dummifier(object):
    
    def __init__(self, output_cols = None, **kwargs):
        self.output_cols = output_cols
        self.kwargs = kwargs
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        dummies = get_dummies(data, **(self.kwargs))
        if self.output_cols:
            return dummies[self.output_cols]
        else:
            return dummies

class CategoricalFilter(object):
    
    def __init__(self, top_categories = 999):
        self.top_categories = top_categories
    
    def fit(self, series):
        counts = series.value_counts()
        self.category_mapper = dict(zip(counts.index[:self.top_categories],
                                    range(1, self.top_categories + 1)))
    
    def transform(self, series):
        return series.apply(lambda key: self.category_mapper.get(key, 0))
        
class FeaturesDummifier(object):
    
    def __init__(self):
        self.colnames = ['feature_' + x.replace(' ', '_') for x in features.FEATURES_MAP.keys()]
    
    def fit(self, data):
        pass
    
    def transform(self, series):
        return DataFrame(features.get_dummies_from_features(series),
                                columns = self.colnames)

class LogTransform(object):
    
    def __init__(self, cols = None):
        self.cols = cols
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        if self.cols:
            data[self.cols] = data[self.cols].applymap(lambda x: np.log(x+1))
            return data
        else:
            return data.applymap(lambda x: np.log(x+1))

class GroupByAggregate(object):
    
    def __init__(self, by, aggregator, reset_index = True):
        self.by = by
        self.aggregator = aggregator
        self.reset_index = reset_index
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        result = data.groupby(self.by).aggregate(self.aggregator)
        if self.reset_index:
            return result.reset_index()
        else:
            return result

class FillNA(object):
    
    def __init__(self, fill_value = 0.0):
        self.fill_value = fill_value
    
    def fit(self, data):
        pass
    
    def transform(self, data):
        return data.fillna(self.fill_value)

class SeparateKey(object):
    
    def __init__(self, key_to_separate):
        self.key = key_to_separate
    
    def __call__(self, data):
        if self.key in data:
            y = data[self.key]
            del data[self.key]
            return data, y
        else:
            return data
