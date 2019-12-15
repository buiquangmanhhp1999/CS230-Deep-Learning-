import numpy as np
import h5py
import matplotlib.pyplot as plt


def sigmoid(z):
    """
        Implements the sigmoid activation in numpy

        Arguments:
        Z -- numpy array of any shape

        Returns:
        A -- output of sigmoid(z), same shape as Z
        cache -- returns Z as well, useful during backpropagation
    """
    A = 1 / (1 + np.exp(-z))
    cache = z
    return A, cache


def relu(z):
    """
        Implement the RELU function.

        Arguments:
        Z -- Output of the linear layer, of any shape

        Returns:
        A -- Post-activation parameter, of the same shape as Z
        cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    A = np.maximum(0, z)
    cache = z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameter_deep(layer_dim):
    """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    L = len(layer_dim)
    parameter = {}

    for i in range(1, L):
        parameter['W' + str(i)] = np.random.randn(layer_dim[i], layer_dim[i - 1]) * 0.01
        parameter['b' + str(i)] = np.zeros((layer_dim[i], 1))

    return parameter


def linear_forward(A, W, b):
    """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = W.dot(A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    linear_cache = (A, W, b)

    return Z, linear_cache


def linear_activation_forward(A_prev, W, b, activation):
    """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
    """

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameter):
    """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    L = len(parameter) // 2
    A = X

    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameter['W' + str(i)], parameter['b' + str(i)],
                                             activation='relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameter['W' + str(L)], parameter['b' + str(L)], activation='sigmoid')
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


def compute_cost(AL, y):
    """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
    """
    m = AL.shape[1]

    cost = (-np.dot(y, np.log(AL.T)) - np.dot(1 - y, np.log(1 - AL.T))) / m

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the back propagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation="sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def random_mini_batches(X, Y, mini_batch_size, seed=0):
    """
        Creates a list of random mini batches from(X, Y)
        :param X: input data, of shape(input size number of examples)
        :param Y: true ' label' vector
        :param mini_batch_size: size of the mini-batches, integer
        :param seed:
        :return: mini-batches: list of synchronous (mini_batch_X, mini_bacth_Y)
    """

    m = X.shape[1]  # number of example
    num_epoch_complete = m // mini_batch_size
    mini_batches = []

    np.random.seed(seed)
    list_shuffled = list(np.random.permutation(m))
    shuffled_X = X[:, list_shuffled]
    shuffled_Y = Y[:, list_shuffled].reshape((1, m))

    for i in range(num_epoch_complete):
        mini_batch_X = shuffled_X[:, i * mini_batch_size: (i + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, i * mini_batch_size: (i + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % num_epoch_complete != 0:
        mini_batch_X = shuffled_X[:, num_epoch_complete * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_epoch_complete * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_parameter_adam(parameters):
    """
        Initialize v and s as two python dictionaries with:
        - keys: 'dW1', 'db1', ..., 'dWL', 'dbL'
        - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameter
        :param parameters: python dictionary containing your parameters.
                            parameters['W' + str(l)] = W1
                            parameters['b' + str(l)] = b1
        :return:
        v -- python dictionary that will contain the exponentially weighted average of the gradient.
        s -- python dictionary that will contain the exponentially weighted average of the squared gradient
    """

    L = len(parameters) // 2
    V = {}
    S = {}
    for i in range(1, L + 1):
        V['dW' + str(i)] = np.zeros_like(parameters['W' + str(i)])
        V['db' + str(i)] = np.zeros_like(parameters['b' + str(i)])
        S['dW' + str(i)] = np.zeros_like(parameters['W' + str(i)])
        S['db' + str(i)] = np.zeros_like(parameters['b' + str(i)])

    return V, S


def update_parameter(parameters, grads, v, s, t, epochs, learning_rate=0.01, b1=0.9, b2=0.999, espilon=1e-8):
    """
        Update parameters using adam algorithm

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters
                      parameters["W" + str(l)] = ...
                      parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # numbers of layers
    v_corrected = {}
    s_corrected = {}
    # perform adam update on all parameters
    for l in range(1, L + 1):
        # Moving average of the gradients. Inputs: 'v, grads, beta1'. Output: 'v'
        v['dW' + str(l)] = b1 * v['dW' + str(l)] + (1 - b1) * grads['dW' + str(l)]
        v['db' + str(l)] = b1 * v['db' + str(l)] + (1 - b1) * grads['db' + str(l)]

        # Moving average of the squared gradients. Inputs: 's, grads, beta2'. Output: 's'
        s['dW' + str(l)] = b2 * s['dW' + str(l)] + (1 - b2) * grads['dW' + str(l)] ** 2
        s['db' + str(l)] = b2 * s['db' + str(l)] + (1 - b2) * grads['db' + str(l)] ** 2

        # Compute bias-corrected first moment estimate. Inputs: 'v, beta1, t'. Output: 'v_corrected'
        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - b1 ** t)
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - b1 ** t)

        # Compute bias-corrected second raw moment estimate,
        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - b2 ** t)
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - b2 ** t)

        # Update parameters.
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * v_corrected['dW' + str(l)] / np.sqrt(
            s_corrected['dW' + str(l)] + espilon)
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * v_corrected['db' + str(l)] / np.sqrt(
            (s_corrected['db' + str(l)] + espilon))

    return parameters, v, s


def L_layer_model(X, Y, layers_dim, learning_rate=0.0007, b1=0.9, b2=0.999, espilon=1e-8, mini_batch_size=64,
                  num_epochs=1200, print_cost=True):
    """
        :param X: input data
        :param Y: true 'label' vector
        :param layers_dim: python list, containing the size of each layer
        :param optimizer:
        :param learning_rate: the learning rate, scalar
        :param mini_batch_size: the size of a mini batch
        :param beta: momentum hyper parameter
        :param beta1: exponential decay hyper parameter for the past gradients estimates
        :param beta2: exponential decay hyper parameter for the past squared gradients estimates
        :param epsilon:
        :param num_epochs: number of epochs
        :param print_cost: True to print the cost every 100 epochs
        :return:
        :parameter: python dictionary containing your updated parameters
    """
    L = len(layers_dim)
    costs = []
    t = 0
    seed = 10
    m = X.shape[1]  # number of examples 209
    parameters = initialize_parameter_deep(layers_dim)
    v, s = initialize_parameter_adam(parameters)

    for i in range(num_epochs):
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        seed = seed + 1
        cost_total = 0

        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            cost_total += compute_cost(AL, mini_batch_Y)
            t = t + 1
            parameters, v, s = update_parameter(parameters, grads, v, s, t, i, learning_rate, b1, b2, espilon)

        cost_avg = cost_total / m
        if print_cost and i % 100 == 0:
            print('Cost %f after %d epochs' % (cost_avg, i))
            costs.append(cost_avg)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))
