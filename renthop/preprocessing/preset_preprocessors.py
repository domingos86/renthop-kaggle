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
    preprocessor.with_pipeline('photo_stats').set_loader(photo_url_merger)
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
    
