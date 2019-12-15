import numpy as np
import matplotlib.pyplot  as plt
import sklearn
import sklearn.datasets
from Improving_Deep_Neural_Networks.Assignment1.init_utils import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = 'he'):
    """
    Implement a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID

    :param X: input data
    :param Y: true label vector
    :param learning_rate: learning rate of gradient descent
    :param num_iterations: number of iterations to run gradient descent
    :param print_cost: if True, print the cost every 1000 iterations
    :param initialization: flag to choose which initialization to use
    :return:
    :parameter: parameters learn by the model
    """

    grads = {}
    costs = []
    m = X.shape[1] # number of examples
    layer_dims = [X.shape[0], 10, 5, 1]

    # Initialization parameters dictionary
    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layer_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layer_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
        a3, cache = forward_propagation(X, parameters)

        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation
        grads = backward_propagation(X, Y, cache)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print('Cost after iteration {}: {}'.format(i, cost))
            costs.append(cost)

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundred)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters

# initialize_parameters_zeros

def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    :param layers_dims: python array(list) containing the size of each layer
    :return: python dictionary containing your parameters "w1", "b1",.."WL","bL":
                        W1: weight matrix
                        b1: bias vector
                        ...
                        WL: weight matrix
                        bL: bias vector
    """

    parametes = {}
    L = len(layers_dims)

    for i in range(1, L):
        parametes['W' + str(i)] = np.zeros((layers_dims[i], layers_dims[i-1]))
        parametes['b' + str(i)] = 0
    return parametes


parameters = model(train_X, train_Y, initialization='zeros')
print('On the train set:')
predictions_train = predict(train_X, train_Y, parameters)
print('On the test set:')
predictions_test = predict(test_X, test_Y, parameters)

print('predictions_train = ', predictions_train)
print('predictions_test = ', predictions_test)

plt.title('Model with Zeros initialization')
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
#plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


def initialize_parameters_random(layers_dims):
    """
    :argument:
    :param layers_dims: python array(list) containing the size of each layer.
    :return:
    :parameter: python dictionary containing your parameters "W1", "b1",..,"WL", "bL"
                W1: weight matrix
                b1: bias vector
                ...
    """

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1])*10
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))

    return parameters


parameters = initialize_parameters_random([3, 2, 1])
print('W1 = ', parameters['W1'])
print('b1 = ', parameters['b1'])
print('W2 = ', parameters['W2'])
print('b2 = ', parameters['b2'])

parameters = model(train_X, train_Y, initialization='random')
print('On the train set:')
predictions_train = predict(train_X, train_Y, parameters)
print('On the test set:')
predictions_test = predict(test_X, test_Y, parameters)

print(predictions_train)
print(predictions_test)

plt.title('Mode with large random initialization')
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

def initialize_parameters_he(layers_dims):
    """
    :argument:
    :param layers_dims:  python array(list) containing the size of each layer.
    :return: python dictionary containing your parameters "W1", "b1", ..."WL", "bL"
    """

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1

    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

