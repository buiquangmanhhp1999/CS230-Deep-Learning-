import numpy as np
import matplotlib.pyplot as plt
from Neural_Networks_and_Deep_learning.CatClassifier.library import *
import h5py
import scipy
from PIL import Image
from scipy import ndimage

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print('y = ' + str(train_y[0, index]) + ".It's a " + classes[train_y[0, index]].decode('utf-8') + 'picture.')

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print('Number of training examples: ' + str(m_train))
print('Number of testing examples: ' + str(m_test))
print('Each image is of size: (' + str(num_px) + ',' + str(num_px) + ',3)')
print('train_x_orig shape: ' + str(train_x_orig.shape))
print('train_y shape: ' + str(train_y.shape) )
print('test_x_orig shape: ' + str(test_x_orig.shape))
print('test_y shape: ' + str(test_y.shape))

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# standardize data to have feature values between 0 and 1
train_x = train_x_flatten / 255  # 12288 * 209
test_x = test_x_flatten / 255

print('train_x shape: ', train_x.shape)
print('test_x shape: ', test_x.shape)
print('train_y shape: ', train_y.shape)  # 1* 209

layers_dims = [train_x.shape[0], 20, 7, 5, 1]
parameter = L_layer_model(train_x, train_y, layers_dims)
predict_train = predict(train_x, train_y, parameter)
predict_test = predict(test_x, test_y, parameter)


