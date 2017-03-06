from sklearn import preprocessing
import numpy as np
import loaders

def features_sentiment_preprocessor():
    json_loader = loaders.JSONLoader()
    preprocessor = loaders.Preprocessor()
    preprocessor.with_pipeline('main').set_loader(json_loader)
    preprocessor.add_operation(loaders.DateTimeExtractor()).add_operation(loaders.NewSimplePredictors())
    preprocessor.add_operation(loaders.LogTransform(['price_per_bedroom', 'price', 'price_per_bathroom']))
    preprocessor.add_operation(loaders.Selector(['listing_id', 'bathrooms', u'bedrooms', 'latitude', 'longitude', 'price',
                                                 'month', 'day_of_month', 'hour', 'day_of_week', 'price_per_bathroom',
                                                 'price_per_bedroom', 'num_features', 'features_len', 'num_photos']))
    merger = loaders.PandasColumnMerger(['main', 'features', 'sentiment'], on = 'listing_id')
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('features').set_loader(loaders.CSVLoader('data/features_train.csv', 'data/features_test.csv'))
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('sentiment').set_loader(loaders.CSVLoader('data/sentiment_train.csv', 'data/sentiment_test.csv'))
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('merged').set_loader(merger).add_operation(loaders.ColumnDrop('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray()).add_operation(preprocessing.StandardScaler())
    preprocessor.with_pipeline('response').set_loader(json_loader.select_loader('interest_level'), only_train = True)
    preprocessor.add_operation(loaders.Dummifier(output_cols = ['high', 'medium', 'low'])).add_operation(loaders.ToNdarray())
    preprocessor.with_pipeline('ids').set_loader(json_loader.select_loader('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray(dtype = np.int64))
    return preprocessor

def features_sentiment_manager_preprocessor():
    json_loader = loaders.JSONLoader()
    preprocessor = loaders.Preprocessor()
    preprocessor.with_pipeline('main').set_loader(json_loader)
    preprocessor.add_operation(loaders.DateTimeExtractor()).add_operation(loaders.NewSimplePredictors())
    preprocessor.add_operation(loaders.LogTransform(['price_per_bedroom', 'price', 'price_per_bathroom']))
    preprocessor.add_operation(loaders.Selector(['listing_id', 'bathrooms', u'bedrooms', 'latitude', 'longitude', 'price',
                                                 'month', 'day_of_month', 'hour', 'day_of_week', 'price_per_bathroom',
                                                 'price_per_bedroom', 'num_features', 'features_len', 'num_photos']))
    merger = loaders.PandasColumnMerger(['main', 'features', 'sentiment'], on = 'listing_id')
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('features').set_loader(loaders.CSVLoader('data/features_train.csv', 'data/features_test.csv'))
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('sentiment').set_loader(loaders.CSVLoader('data/sentiment_train.csv', 'data/sentiment_test.csv'))
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('merged').set_loader(merger).add_operation(loaders.ColumnDrop('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray()).add_operation(preprocessing.StandardScaler())
    preprocessor.with_pipeline('response').set_loader(json_loader.select_loader('interest_level'), only_train = True)
    preprocessor.add_operation(loaders.Dummifier(output_cols = ['high', 'medium', 'low'])).add_operation(loaders.ToNdarray())
    preprocessor.with_pipeline('managers').set_loader(json_loader.select_loader('manager_id'))
    preprocessor.add_operation(loaders.CategoricalFilter(999)).add_operation(loaders.ToNdarray(dtype = np.int64, outshape = (-1, 1)))
    preprocessor.with_pipeline('ids').set_loader(json_loader.select_loader('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray(dtype = np.int64))
    return preprocessor

def features_sentiment_manager_sharpness_preprocessor():
    json_loader = loaders.JSONLoader()
    preprocessor = loaders.Preprocessor()
    preprocessor.with_pipeline('origin').set_loader(json_loader)
    preprocessor.add_operation(loaders.DateTimeExtractor()).add_operation(loaders.NewSimplePredictors())
    preprocessor.add_operation(loaders.LogTransform(['price_per_bedroom', 'price', 'price_per_bathroom']))
    preprocessor.add_operation(loaders.Selector(['listing_id', 'bathrooms', u'bedrooms', 'latitude', 'longitude', 'price',
                                                 'month', 'day_of_month', 'hour', 'day_of_week', 'price_per_bathroom',
                                                 'price_per_bedroom', 'num_features', 'features_len', 'num_photos']))
    merger = loaders.PandasColumnMerger(['origin', 'features', 'sentiment', 'photo_stats'], on = 'listing_id')
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('features').set_loader(loaders.CSVLoader('data/features_train.csv', 'data/features_test.csv'))
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('sentiment').set_loader(loaders.CSVLoader('data/sentiment_train.csv', 'data/sentiment_test.csv'))
    preprocessor.set_consumer(merger)
    photo_url_merger = loaders.GetTopPhotoMerger('photo_stats_sharpness', 'photo_stats_photo_url')
    preprocessor.with_pipeline('photo_stats_sharpness').set_loader(loaders.CSVLoader('data/images_train.csv', 'data/images_test.csv'))
    preprocessor.set_consumer(photo_url_merger)
    preprocessor.with_pipeline('photo_stats_photo_url').set_loader(json_loader.select_loader(['listing_id', 'photos']))
    preprocessor.set_consumer(photo_url_merger)
    preprocessor.with_pipeline('photo_stats').set_loader(photo_url_merger).add_operation(loaders.ColumnDrop('photo_name'))
    preprocessor.add_operation(loaders.LogTransform(['avg_sharpness', 'cover_sharpness'])).set_consumer(merger)
    preprocessor.with_pipeline('main').set_loader(merger).add_operation(loaders.ColumnDrop('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray()).add_operation(preprocessing.StandardScaler())
    preprocessor.with_pipeline('response').set_loader(json_loader.select_loader('interest_level'), only_train = True)
    preprocessor.add_operation(loaders.Dummifier(output_cols = ['high', 'medium', 'low'])).add_operation(loaders.ToNdarray())
    preprocessor.with_pipeline('managers').set_loader(json_loader.select_loader('manager_id'))
    preprocessor.add_operation(loaders.CategoricalFilter(999)).add_operation(loaders.ToNdarray(dtype = np.int64, outshape = (-1, 1)))
    preprocessor.with_pipeline('ids').set_loader(json_loader.select_loader('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray(dtype = np.int64))
    return preprocessor
    
def cover_features_sentiment_manager_sharpness_preprocessor():
    '''
    Returns pipelines main (83 fields), managers (1 field), photo_cover_stats (3 fields), photo_cover (3*100*300), ids (1 field), response (3 fields)
    '''
    json_loader = loaders.JSONLoader(sample = 20000)
    preprocessor = loaders.Preprocessor()
    preprocessor.with_pipeline('origin').set_loader(json_loader)
    preprocessor.add_operation(loaders.DateTimeExtractor()).add_operation(loaders.NewSimplePredictors())
    preprocessor.add_operation(loaders.LogTransform(['price_per_bedroom', 'price', 'price_per_bathroom']))
    preprocessor.add_operation(loaders.Selector(['listing_id', 'bathrooms', u'bedrooms', 'latitude', 'longitude', 'price',
                                                 'month', 'day_of_month', 'hour', 'day_of_week', 'price_per_bathroom',
                                                 'price_per_bedroom', 'num_features', 'features_len', 'num_photos']))
    merger = loaders.PandasColumnMerger(['origin', 'features', 'sentiment', 'photo_stats'], on = 'listing_id', how = 'left')
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('features').set_loader(loaders.CSVLoader('data/features_train.csv', 'data/features_test.csv'))
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('sentiment').set_loader(loaders.CSVLoader('data/sentiment_train.csv', 'data/sentiment_test.csv'))
    preprocessor.set_consumer(merger)
    photo_url_merger = loaders.GetTopPhotoMerger('photo_stats_sharpness', 'photo_stats_photo_url')
    preprocessor.with_pipeline('photo_stats_sharpness').set_loader(loaders.CSVLoader('data/images_train.csv', 'data/images_test.csv'))
    preprocessor.set_consumer(photo_url_merger)
    preprocessor.with_pipeline('photo_stats_photo_url').set_loader(json_loader.select_loader(['listing_id', 'photos']))
    preprocessor.set_consumer(photo_url_merger)
    preprocessor.with_pipeline('photo_stats').set_loader(photo_url_merger)
    preprocessor.add_operation(loaders.Selector(['listing_id', 'avg_width', 'avg_height', 'avg_sharpness']))
    preprocessor.add_operation(loaders.LogTransform(['avg_sharpness'])).set_consumer(merger)
    preprocessor.with_pipeline('main').set_loader(merger).add_operation(loaders.ColumnDrop('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray()).add_operation(preprocessing.StandardScaler())
    preprocessor.with_pipeline('response').set_loader(json_loader.select_loader('interest_level'), only_train = True)
    preprocessor.add_operation(loaders.Dummifier(output_cols = ['high', 'medium', 'low'])).add_operation(loaders.ToNdarray())
    preprocessor.with_pipeline('managers').set_loader(json_loader.select_loader('manager_id'))
    preprocessor.add_operation(loaders.CategoricalFilter(999)).add_operation(loaders.ToNdarray(dtype = np.int64, outshape = (-1, 1)))
    preprocessor.with_pipeline('photo_cover_stats').set_loader(photo_url_merger)
    preprocessor.add_operation(loaders.Selector(['cover_width', 'cover_height', 'cover_sharpness']))
    preprocessor.add_operation(loaders.LogTransform(['cover_sharpness']))
    preprocessor.add_operation(loaders.ToNdarray()).add_operation(preprocessing.StandardScaler())
    preprocessor.with_pipeline('photo_cover_gather').set_loader(photo_url_merger)
    preprocessor.add_operation(loaders.Selector(['listing_id', 'photo_name']))
    photo_loader = loaders.PhotoLoader() # using a loader in order to make different transformations depending on whether test or train
    preprocessor.set_consumer(photo_loader)
    preprocessor.with_pipeline('photo_cover').set_loader(photo_loader)
    preprocessor.with_pipeline('ids').set_loader(json_loader.select_loader('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray(dtype = np.int64))
    return preprocessor

def activations_features_sentiment_manager_sharpness_preprocessor():
    '''
    Returns pipelines main (103 fields), managers (1 field), ids (1 field), response (3 fields)
    '''
    json_loader = loaders.JSONLoader()
    preprocessor = loaders.Preprocessor()
    preprocessor.with_pipeline('origin').set_loader(json_loader)
    preprocessor.add_operation(loaders.DateTimeExtractor()).add_operation(loaders.NewSimplePredictors())
    preprocessor.add_operation(loaders.LogTransform(['price_per_bedroom', 'price', 'price_per_bathroom']))
    preprocessor.add_operation(loaders.Selector(['listing_id', 'bathrooms', u'bedrooms', 'latitude', 'longitude', 'price',
                                                 'month', 'day_of_month', 'hour', 'day_of_week', 'price_per_bathroom',
                                                 'price_per_bedroom', 'num_features', 'features_len', 'num_photos']))
    merger = loaders.PandasColumnMerger(['origin', 'features', 'sentiment', 'photo_stats', 'cover_activations'],
                                        on = 'listing_id', how = 'left')
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('features').set_loader(loaders.CSVLoader('data/features_train.csv', 'data/features_test.csv'))
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('sentiment').set_loader(loaders.CSVLoader('data/sentiment_train.csv', 'data/sentiment_test.csv'))
    preprocessor.set_consumer(merger)
    photo_url_merger = loaders.GetTopPhotoMerger('photo_stats_sharpness', 'photo_stats_photo_url')
    preprocessor.with_pipeline('photo_stats_sharpness').set_loader(loaders.CSVLoader('data/images_train.csv', 'data/images_test.csv'))
    preprocessor.set_consumer(photo_url_merger)
    preprocessor.with_pipeline('photo_stats_photo_url').set_loader(json_loader.select_loader(['listing_id', 'photos']))
    preprocessor.set_consumer(photo_url_merger)
    preprocessor.with_pipeline('photo_stats').set_loader(photo_url_merger)
    preprocessor.add_operation(loaders.Selector(['listing_id', 'avg_width', 'avg_height', 'avg_sharpness']))
    preprocessor.add_operation(loaders.LogTransform(['avg_sharpness'])).set_consumer(merger)
    preprocessor.with_pipeline('cover_activations').set_loader(loaders.CSVLoader('data/images_activations_train.csv',
                                                                                 'data/images_activations_test.csv'))
    preprocessor.set_consumer(merger)
    preprocessor.with_pipeline('main').set_loader(merger).add_operation(loaders.ColumnDrop('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray()).add_operation(preprocessing.StandardScaler())
    preprocessor.with_pipeline('response').set_loader(json_loader.select_loader('interest_level'), only_train = True)
    preprocessor.add_operation(loaders.Dummifier(output_cols = ['high', 'medium', 'low'])).add_operation(loaders.ToNdarray())
    preprocessor.with_pipeline('managers').set_loader(json_loader.select_loader('manager_id'))
    preprocessor.add_operation(loaders.CategoricalFilter(999)).add_operation(loaders.ToNdarray(dtype = np.int64, outshape = (-1, 1)))
    preprocessor.with_pipeline('ids').set_loader(json_loader.select_loader('listing_id'))
    preprocessor.add_operation(loaders.ToNdarray(dtype = np.int64))
    return preprocessor
