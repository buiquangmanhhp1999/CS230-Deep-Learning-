import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def zero_pad(X, pad):
    """
    Pad with zeros all image of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1
    :param X: python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    :param pad: integer, amount of padding around each image on vertical and horizontal dimensions
    :return: X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    x_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))

    return x_pad


np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print('x.shape = \n', x.shape)
print('x_pad.shape =\n', x_pad.shape)
print('x[1, 1] =\n', x[1, 1].shape)
print('x_pad[1, 1] =\n', x_pad[1, 1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0, :, :, 0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.
    :param a_slice_prev: slice of input data of shape (f, f, n_C_prev)
    :param W: Weight parameters contained in  a window - matrix of shape (f, f, n_C_prev)
    :param b: Bias parameters contained in a window - matrix of shape (1, 1, 1)
    :return:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice_prev and W. Do not add the bias yet
    s = a_slice_prev * W
    # Sum over all entries of the volume s. Then add bias b to Z.Cast b to a float() so that Z results in a scalar value
    z = np.sum(s) + float(b)
    return z


np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print('Z = ', Z)


def conv_forward(A_prev, W, b, hpatameters):
    """
    Implementss the forward propagation for a convolution function
    :param A_prev: output activations of th previous layer
                    numpy array of shape (m, n_H-prev, n_W_prev, n_C_prev)
    :param W: Weight, numpy array of shape (f, f, n_C_prev, n_C)
    :param b: Biases, numpy array of shape (1, 1, 1, n_C)
    :param hpatameters: python dictionary containing 'stride' and 'pad'
    :return:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimension from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimension from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from 'hparameters'
    stride = hpatameters['stride']
    pad = hpatameters['pad']

    # Compute the dimension of the CONV output volume using the formula given above
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    for i in range(m):  # Loop over the batch of training example
        a_prev_pad = A_prev_pad[i]  # Select ith training example's padded activation
        for h in range(n_H):    # Loop over vertical axis of the output volume
            # Find the vertical start and end of the current 'slice'
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):    # Loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current slice
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):    # Loop over channels
                    # Use the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias
                    weights = W[..., c]
                    biases = b[..., c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    # Making sure your output shape is correct
    assert (Z.shape == (m, n_H, n_W, n_C))

    # Save information in 'cache' for the backprop
    cache = (A_prev, W, b, hpatameters)

    return Z, cache


np.random.seed(1)
A_prev = np.random.randn(10, 5, 7, 4)
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad": 1, 'stride': 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =\n", np.mean(Z))
print("Z[3,2,1] =\n", Z[3,2,1])
print("cache_conv[0][1][2][3] =\n", cache_conv[0][1][2][3])


def pool_forward(A_prev, hparameters, mode='max'):
    """
    Implements the forward pass of the pooling layer
    :param A_prev: Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param hparameters: python dictionary containing 'f' and 'stride'
    :param mode: the pooling mode you would like to use, defined as a string ('max' or 'average')
    :return:
    A - output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """

    # Retrieve dimension from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters['f']
    stride = hparameters['stride']

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize the dimensions of the output
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):          # Loop over the training examples
        a_prev = A_prev[i]
        for h in range(n_H):    # Loop on the vertical axis
            # find the vertical start and end of the current 'slice'
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):    # Loop on the horizontal axis
                # Find the vertical start and end of the current slice
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):    # Loop over the channels of the output
                    # Use the corners to define the current slice on the ith training examples
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    # Compute the pooling operation on the slice
                    # Use an if statement to differentiate the model
                    # Use np.max and np.mean
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    # Store the input and hparameters in 'cache' for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache


# Case 1: stride of 1
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {'stride': 1, 'f': 3}

A, cache = pool_forward(A_prev, hparameters)
print('mode = max')
print('A.shape = ' + str(A.shape))
print('A = \n', A)
print()
A, cache = pool_forward(A_prev, hparameters, mode='average')
print('mode = average')
print('A.shape = ' + str(A.shape))
print('A = \n', A)

# Case 2: stride of 2
np.random.seed(1)
A_prev = np.random.randn(2, 5, 5, 3)
hparameters = {'stride': 2, 'f': 3}

A, cache = pool_forward(A_prev, hparameters)
print('mode = max')
print('A.shape = ' + str(A.shape))
print('A = \n', A)
print()

A, cache = pool_forward(A_prev, hparameters, mode = 'average')
print('mode = average')
print('A.shape = '+ str(A.shape))
print('A = \n', A)


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x
    :param x: Array of shape (f, f)
    :return:
    mask -- Array of the same shape as wondpw, contains a True at the position corresponding to the max entry of x
    """
    mask = (x == np.max(x))
    return mask


np.random.seed(1)
x = np.random.randn(2, 3)
mask = create_mask_from_window(x)
print('x = ', x)
print('mask = ', mask)


def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    :param dz: input scalar
    :param shape: the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    :return:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """

    # Retrieve dimension from shape
    (n_H, n_W) = shape

    # Compute the value to distribute on the matrix
    average = np.ones((n_H, n_W))

    # Create a matrix where every entry is the 'average' value
    a = dz * average / np.sum(average)
    return a


a = distribute_value(2, (2, 2))
print('distribute value =\n', a)






