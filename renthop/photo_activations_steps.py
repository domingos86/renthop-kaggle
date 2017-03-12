import numpy as np
import pandas as pd
import re, sys
from renthop.preprocessing import preset_preprocessors
from renthop.nn_model import convolutional

# train cover photo model
preprocessor = preset_preprocessors.cover_features_sentiment_manager_sharpness_preprocessor()
data = preprocessor.load_and_transform()
y = data['response']
del data['response']
del data['ids']
net, history = convolutional.fit(data, y, epochs = 500, save_to = 'nn_trained/net4')

# save more model details
preprocessor.save('nn_trained/net4/prep.pickle')
convolutional.plot_net(history, save_to = 'nn_trained/net4/net.png')

# recover convolutional section of model
net_best = convolutional.load_net('nn_trained/net4/net.h5')
conv_net = net_best.layers[0].layers[2]
cover_sharpness_scaler = preprocessor.operations_photo_cover_stats[3]
from renthop.preprocessing import loaders
photo_loader = loaders.PhotoLoader()

# find activations for train data
image_data = pd.read_csv('data/images_train.csv')
data_train = pd.read_json('data/train.json')
data_train = data_train[['listing_id', 'photos']]
data_train['photo_name'] = data_train['photos'].apply(lambda x: x[0] if len(x) > 0 else '').apply(lambda x: re.sub(r'^.*/', '', x))
data_train = data_train.drop('photos', axis = 1)
data_train = data_train.merge(image_data[['listing_id', 'photo_name', 'width', 'height', 'sharpness']], how = 'left', on = ['listing_id', 'photo_name'])
data_train = data_train.fillna(0.0)
photo_names = data_train[['listing_id', 'photo_name']]
data_train['sharpness'] = np.log(data_train['sharpness'] + 1)
photo_stats = np.array(data_train[['width', 'height', 'sharpness']], dtype = np.float32)
photo_stats = cover_sharpness_scaler.transform(photo_stats)
activations = np.zeros((0, 20), dtype = np.float32)
for i in range(0, photo_names.shape[0], 1000):
    X_ = {}
    X_['photo_cover_stats'] = photo_stats[i:(i + 1000)]
    photo_loader.consume(photo_names.iloc[i:(i + 1000), :], None)
    X_['photo_cover'] = photo_loader()
    activations = np.vstack((activations, conv_net.predict(X_)))
    sys.stdout.write('.')
    sys.stdout.flush()
print ""
df = pd.concat([photo_names[['listing_id']], pd.DataFrame(activations)], axis = 1)
df.to_csv('data/images_activations_train.csv', index = False)

# find activations for test data
data_test = pd.read_json('data/test.json')
image_data = pd.read_csv('data/images_test.csv')
data_test = data_test[['listing_id', 'photos']]
data_test['photo_name'] = data_test['photos'].apply(lambda x: x[0] if len(x) > 0 else '').apply(lambda x: re.sub(r'^.*/', '', x))
data_test = data_test.drop('photos', axis = 1)
data_test = data_test.merge(image_data[['listing_id', 'photo_name', 'width', 'height', 'sharpness']], how = 'left', on = ['listing_id', 'photo_name'])
data_test = data_test.fillna(0.0)
photo_names = data_test[['listing_id', 'photo_name']]
data_test['sharpness'] = np.log(data_test['sharpness'] + 1)
photo_stats = np.array(data_test[['width', 'height', 'sharpness']], dtype = np.float32)
photo_stats = cover_sharpness_scaler.transform(photo_stats)
activations = np.zeros((0, 20), dtype = np.float32)
for i in range(0, photo_names.shape[0], 1000):
    X_ = {}
    X_['photo_cover_stats'] = photo_stats[i:(i + 1000)]
    photo_loader.consume(photo_names.iloc[i:(i + 1000), :], None)
    X_['photo_cover'] = photo_loader()
    activations = np.vstack((activations, conv_net.predict(X_)))
    sys.stdout.write('.')
    sys.stdout.flush()
print ""
df = pd.concat([photo_names[['listing_id']], pd.DataFrame(activations)], axis = 1)
df.to_csv('data/images_activations_test.csv', index = False)
