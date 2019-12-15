import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from Improving_Deep_Neural_Networks.Assignment2.opt_utils_v1a import *
from Improving_Deep_Neural_Networks.Assignment2.testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    :param parameters: python dictionary containing your parameters to be updated
    :param grads: python dictionary containing your gradients to update each parameters
    :param learning_rate: the learning rate, scalar
    :return:
    parameters: python dictionary containing your updated parameters

    """

    L = len(parameters) // 2 #  number of layers in the neural networks
    for l in range(1, L + 1):
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * grads['db' + str(l)]

    return parameters


parameters, grads, learning_rate = update_parameters_with_gd_test_case()

parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print("W1 =\n" + str(parameters["W1"]))
print("b1 =\n" + str(parameters["b1"]))
print("W2 =\n" + str(parameters["W2"]))
print("b2 =\n" + str(parameters["b2"]))


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random mini batches from(X, Y)
    :param X: input data, of shape(input size number of examples)
    :param Y: true ' label' vector
    :param mini_batch_size: size of the mini-batches, integer
    :param seed:
    :return: mini-batches: list of synchronous (mini_batch_X, mini_bacth_Y)
    """

    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffle_X, shuffled_Y). Minus the end case
    num_complete_minibatches = m // mini_batch_size     # number of mini batches of size mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[0, k * mini_batch_size : (k+1) * mini_batch_size].reshape((1, -1))
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-bathc < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[0, num_complete_minibatches * mini_batch_size:].reshape((1, -1))
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))


def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: 'dW1', 'db1', ..., 'dWL', 'dbL'
                - values: numpy arrays  pf zeros of the same shape as the corresponding gradients/parameters
    Arguments:
    :param parameters:  python dictionary containing the current velocity
                        v['dW' + str(l+1)] = velocity of dW1
                    v['db' + str(l)] = velocity of db1
    :return:
    v -- python dictionary containing the current velocity
            v['dW' + str(l)] = velocity of dW1
            v['db' + str(l)] = velocity of db1
    """

    L = len(parameters) // 2    # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        v['dW' + str(l+1)] = np.zeros_like(parameters['W' + str(l+1)])
        v['db' + str(l+1)] = np.zeros_like(parameters['b' + str(l+1)])

    return v


parameters = initialize_velocity_test_case()

v = initialize_velocity(parameters)
print("v[\"dW1\"] =\n" + str(v["dW1"]))
print("v[\"db1\"] =\n" + str(v["db1"]))
print("v[\"dW2\"] =\n" + str(v["dW2"]))
print("v[\"db2\"] =\n" + str(v["db2"]))


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    :param parameters: python dictionary containing your parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = b1
    :param grads: python dictionary containing your gradients for each parameters:
    :param v: python dictionary containing the current velocity:
                v['dW' + str(l)] = ..
                v['db' + str(l)] = ...
    :param beta: the momentum hyper parameter, scalar
    :param learning_rate: the learning rate, scalar
    :return:
    parameters: python dictionary containing your updated parameters
    v: python dictionary containing your updated velocities

    """

    L = len(parameters) // 2        # number of layers in the neural networks

    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v['dW' + str(l + 1)] = beta * v['dW' + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta * v['db' + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
        # update parameters
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v['db' + str(l + 1)]

    return parameters, v


parameters, grads, v = update_parameters_with_momentum_test_case()

parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
print("W1 = \n" + str(parameters["W1"]))
print("b1 = \n" + str(parameters["b1"]))
print("W2 = \n" + str(parameters["W2"]))
print("b2 = \n" + str(parameters["b2"]))
print("v[\"dW1\"] = \n" + str(v["dW1"]))
print("v[\"db1\"] = \n" + str(v["db1"]))
print("v[\"dW2\"] = \n" + str(v["dW2"]))
print("v[\"db2\"] = v" + str(v["db2"]))


def initialize_adam(parameters):
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

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    # Initialize v, s. Input: 'parameters'. outputs: 'v, s'
    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros_like(parameters['W' + str(l + 1)])
        v['db' + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])
        s['dW' + str(l + 1)] = np.zeros_like(parameters['W' + str(l + 1)])
        s['db' + str(l + 1)] = np.zeros_like(parameters['b' + str(l + 1)])

    return v, s


parameters = initialize_adam_test_case()

v, s = initialize_adam(parameters)
print("v[\"dW1\"] = \n" + str(v["dW1"]))
print("v[\"db1\"] = \n" + str(v["db1"]))
print("v[\"dW2\"] = \n" + str(v["dW2"]))
print("v[\"db2\"] = \n" + str(v["db2"]))
print("s[\"dW1\"] = \n" + str(s["dW1"]))
print("s[\"db1\"] = \n" + str(s["db1"]))
print("s[\"dW2\"] = \n" + str(s["dW2"]))
print("s[\"db2\"] = \n" + str(s["db2"]))


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    """
    Update parameters using Adam
    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients for each parameters
    :param v: Adam variable, moving average of the first gradient, python dictionary
    :param s: Adam variable. moving average of the squared gradient, python dictionary
    :param t:
    :param learning_rate: the learning rate, scalar
    :param beta1: Exponential decay hyper parameter for the first moment estimates
    :param beta2: Exponential decay hyper parameter for the second moment estimates
    :param epsilon: hyper parameter preventing division by zero in Adam updates
    :return:
    :parameter: python dictionary containing your updated parameters
    v: adam variable, moving average of the first gradient, python dictionary
    s: adam variable, moving average of the squared gradient, python dictionary

    """

    L = len(parameters) // 2    # number of layers in the neural networks
    v_corrected = {}
    s_corrected = {}

    # perform adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: 'v, grads, beta1'. Output: 'v'
        v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        # Compute bias-corrected first moment estimate. Inputs: 'v, beta1, t'. Output: 'v_corrected'
        v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - beta1**t)
        v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - beta1**t)

        # Moving average of the squared gradients. Inputs: 's, grads, beta2'. Output: 's'
        s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * grads['dW' + str(l + 1)]**2
        s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * grads['db' + str(l + 1)]**2

        # Compute bias-corrected second raw moment estimate,
        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - beta2**t)
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - beta2**t)

        # Update parameters.
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * v_corrected['dW' + str(l + 1)] / (np.sqrt(s_corrected['dW' + str(l + 1)]) + epsilon)
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * v_corrected['db' + str(l + 1)] / (np.sqrt(s_corrected['db' + str(l + 1)]) + epsilon)

    return parameters, v, s


parameters, grads, v, s = update_parameters_with_adam_test_case()
parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)

print("W1 = \n" + str(parameters["W1"]))
print("b1 = \n" + str(parameters["b1"]))
print("W2 = \n" + str(parameters["W2"]))
print("b2 = \n" + str(parameters["b2"]))
print("v[\"dW1\"] = \n" + str(v["dW1"]))
print("v[\"db1\"] = \n" + str(v["db1"]))
print("v[\"dW2\"] = \n" + str(v["dW2"]))
print("v[\"db2\"] = \n" + str(v["db2"]))
print("s[\"dW1\"] = \n" + str(s["dW1"]))
print("s[\"db1\"] = \n" + str(s["db1"]))
print("s[\"dW2\"] = \n" + str(s["dW2"]))
print("s[\"db2\"] = \n" + str(s["db2"]))

train_X, train_Y = load_dataset()


def model(X, Y,layers_dim, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2 =0.999, epsilon=1e-8, num_epochs=10000, print_cost = True):
    """
    3-layer neural network model which can be run in different optimizer models
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
    :param print_cost: True to print the cost every 1000 epochs
    :return:
    :parameter: python dictionary containing your updated parameters
    """

    L = len(layers_dim)     # number of layers in the neuron networks
    costs = []              # to keep track of the cost
    t = 0                   # initializing the counter required for Adam update
    seed = 10
    m = X.shape[1]           # number of training examples

    # Initialize parameters
    parameters = initialize_parameters(layers_dim)

    # Initialize the optimizer
    if optimizer == 'gd':
        pass  # no initialization required for gradient descent
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)
    elif optimizer == 'adam':
        v, s = initialize_adam(parameters)

    # Optimization Loop
    for i in range(num_epochs):
        # Define the random mini-batches
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == 'gd':
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == 'adam':
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
            elif optimizer == 'momentum':
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)

        cost_avg = cost_total / m

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print('Cost after epoch %i: %f' %(i, cost_avg))
            costs.append(cost_avg)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters


# train 3-layer model
print('Train X', train_X.shape)
print('Train Y', train_Y.shape)
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer="gd")

# Predict
predictions = predict(train_X, train_Y, parameters)


# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])


# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer="adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])

# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


