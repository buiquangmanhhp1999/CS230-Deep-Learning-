import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from Neural_Networks_and_Deep_learning.Assignment4.dnn_app_utils_v3 import *


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
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

print('train_x shape: ', train_x.shape)
print('test_x shape: ', test_x.shape)

# two layer model
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR-RELU-LINEAR-SIGMOID.

    :param X: input data
    :param Y: true 'label' vector
    :param layers_dims: dimensions of the layers
    :param learning_rate: learning rate of the gradient descent update rule
    :param num_iterations: number of iterations of the optimization
    :param print_cost: if set to True, this will print the cost every 100 iterations
    :return:
    parameters -- a dictionary containing W1, W2, b1  and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the function you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Get W1, b1, W2, b2 from the dictionary parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Loop (gradient descent)

    for i in range(num_iterations):
        # Forward propagation: LINEAR->RELU->LINEAR->SIGMOID.
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 =  linear_activation_forward(A1, W2, b2, 'sigmoid')

        # Compute cost
        cost = compute_cost(A2, Y)

        # Initializing backward propagation
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        # Set grads
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        if print_cost and i % 100 == 0:
            print('Cost after iteration {}: {}'.format(i, np.squeeze(cost)))
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundred)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters


# parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
# predictions_test = predict(test_x, test_y, parameters)


# L layer model
layers_dims2 = [12288, 20, 7, 5, 1] # 4 layers model


def L_layer_model(X, Y, layers_dims, learning_rate = 0.009, num_iterations = 3000, print_cost=False):
    """
    Implement a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID

    :param X: data
    :param Y: true "label" vector
    :param layers_dims: list containing the input size and each layer size
    :param learning_rate: learning rate of the gradient descent update rule
    :param num_iterations: number of iteratios of the optimization
    :param print_cost: if True, it prints the cost every 100 steps
    :return:
    parameters: parameters learnt by the model. They can then be used to predict
    """

    np.random.seed(1)
    costs = []

    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update propagation
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print('Cost after iteration %i: %f' %(i, cost))
            costs.append(cost)

        # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundred)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


parameters = L_layer_model(train_x, train_y, layers_dims2, num_iterations=2500, print_cost=True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
