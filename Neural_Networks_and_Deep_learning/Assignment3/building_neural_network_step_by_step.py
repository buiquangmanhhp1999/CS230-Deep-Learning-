import numpy as np
import h5py
import matplotlib.pyplot as plt
from Neural_Networks_and_Deep_learning.Assignment3.testCases_v4a import *
from Neural_Networks_and_Deep_learning.Assignment3.dnn_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    """
    :argument
    :param n_x: size of the input layer
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer
    :return:
    :parameter: python dictionary containing your parameters:
                W1: weight matrix of shape (n_h, n_x)
                b1: bias vector of shape (n_h, 1)
                W2: weight matrix of shape (n_y, n_h)
                b2: bias vector of shape (n_y, 1)
    """
    np.random.seed(1)

    W1 = np.random.rand(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def initialize_parameters_deep(layer_dims):
    """
    :param layer_dims: python array (list) containing the dimensions of each layer in our network

    :return:
    :parameter: python dictionary containing your parameters "W1", "b1", ... "WL", "bL"
                W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1]
                b1 -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.rand(layer_dims[l], layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    :param A: activations from previous layer (or input data): (size of previous layer, number of examples)
    :param W: weights matrix
    :param b: bias vector
    :return:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W", and "b"; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_pre, W, b, activation):
    """
    Implement the forward propagation for the Linear->Activation layer
    :argument
    :param A_pre: activations from previous layer (or input data)
    :param W: weights matrix
    :param b: bias vector
    :param activation: the activation to be used in this layer, stored as a text string
    :return:
            A: the output of the activation function, also called the post-activation value
            cache: a python tuple containing "linear_cache" and "activation_cache";
                    stored for computing the backward pass efficiently
    """

    Z, linear_cache = linear_forward(A_pre, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_pre.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the {Linear->RELU]*(L-1)->LINEAR->SIGMOID computation
    :param X: data, numpy array of shape (input size, number of examples
    :param parameters: output of initialize_parameters_deep()
    :return:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward()
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of Layers in the neural network

    for l in range(1, L):
        A_pre = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_pre, W, b, "relu")
        caches.append(cache)

    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A, W, b, "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7)
    :param AL: probability vector corresponding to your label predictions
    :param Y: true "label" vector
    :return: cost -- cross-entropy cost
    """

    m = Y.shape[1]
    cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost

def linear_backward(dZ, cache):
    """
    Imlement the linear portion of backward propagation for a single layer
    :argument
    :param dZ: Gradient of the cost with respect to the linear output (of current layer l)
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    :return:
    dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW: Gradient of the cost with respect to W(current layer l), same shape as W
    db: Gradient of the cost with respect to b(current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_pre = np.dot(W.T, dZ)

    assert (dA_pre.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_pre, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer
    :param dA: post-activation gradient for current layer l
    :param cache: tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    :return:
    dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW: Gradient of the cost with respect to W(current layer l), same shape as W
    db: Gradient of the cost with respect to b(current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L - 1) -> LINEAR -> SIGMOID group
    :param AL: probability vector, output of the forward propagation
    :param Y: true "label" vector
    :param caches: list of caches containing:
                    every cache of linear_activation_forward() with "relu"
                    the cache of linear_activation_forward() with "sigmoid"
    :return:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> Linear) gradients.
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    # Loop from L = L -2 to L = 0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradient, output of L_model_backward
    :param learning_rate:
    :return:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of Layers in the neural network
    # Update rule for each parameter, Use a for Loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = " + str(parameters["W1"]))
print ("b1 = " + str(parameters["b1"]))
print ("W2 = " + str(parameters["W2"]))
print ("b2 = " + str(parameters["b2"]))