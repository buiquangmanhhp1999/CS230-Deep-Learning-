from Neural_Networks_and_Deep_learning.Assignment1.lr_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

num_train = train_set_x_orig.shape[0]
num_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print('Number of training examples: m_train = ' + str(num_train))
print('Number of testing examples: m_test = ' + str(num_test))
print('Height/Width of each image: num_px = ', num_px)
print('train_set_x shape: ', train_set_x_orig.shape)
print('train_set_y shape: ', train_set_y.shape)
print('test_set_x shape: ', test_set_x_orig.shape)
print('test_set_y shape ', test_set_y.shape)

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(num_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(num_test, -1).T

print('Size of training examples flatten: ', train_set_x_flatten.shape)
print('Size of train_set_y: ', train_set_y.shape)
print('Size of testing examples: ', test_set_x_flatten.shape)
print('Size of test_set_y: ', test_set_y.shape)

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

def propagate(w, b, X, Y):
    """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
    """

    num_train = X.shape[1]
    dw = np.zeros_like(w)

    # FORWARD PROPAGATION
    z = w.T.dot(X) + b
    A = sigmoid(z)
    cost = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / (-num_train)

    # BACKWARD PROPAGATION
    dw = X.dot((A - Y).T) / num_train
    db = np.sum(A - Y) / num_train

    '''assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())'''

    grads = {'dw': dw, 'db': db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = None, None
        grads, cost = propagate(w, b, X, Y)

        # retrive derivatives from grads
        dw = grads['dw']
        db = grads['db']

        # update rule
        w = w - learning_rate * dw
        b = b - db

        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print('Cost afer iteration %i: %f' %(i, cost))

    params = {'w': w,
              'b': b}

    grads = {'dw': dw,
             'db': db}

    return params, grads, costs

def predict(w, b, X):
    '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    num_train = X.shape[1]
    Y_prediction = np.zeros((1, num_train))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(w.T.dot(X) + b)
    for i in range(A.shape[1]):
        if A[0, i] < 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    assert (Y_prediction.shape == (1, num_train))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
        Builds the logistic regression model by calling the function you've implemented previously

        Arguments:
        X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
    """

    #initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    #gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    #retrieve parameters w and b
    w = parameters['w']
    b = parameters['b']

    #predict test/train set examples
    y_prediction_test = predict(w, b, X_test)
    y_prediction_train = predict(w, b, X_train)

    print('Train accuracy: {} %'.format(100 - np.mean(np.abs(y_prediction_train - Y_train)) * 100))
    print('Test accuracy: {} %'.format(100 - np.mean(np.abs(y_prediction_test - Y_test)) * 100))

    d = {'costs': costs,
         'Y_predict_test': y_prediction_test,
         'Y_prediction_train': y_prediction_train,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations}

    return d

d= model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)


