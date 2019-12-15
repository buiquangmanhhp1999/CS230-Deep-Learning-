import numpy as np
from keras.layers import *
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from Convolutionnal_Neural_Network.Week2.Assignment1.kt_utils import *
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow

X_train_orig, Y_train_ori, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255
X_test = X_test_orig / 255

# Reshape
Y_train = Y_train_ori.T
Y_test = Y_test_orig.T

print('number of training examples = ' + str(X_train.shape[0]))
print('number of test examples = ' + str(X_test.shape[0]))
print('X_train shape: ' + str(X_train.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test shape : ' + str(X_test.shape))
print('Y_test shape : ' + str(Y_test.shape))


def HappyModel(input_shape):
    """
    Implementation of the HappyModel

    Arguments:
    input_shape -- shape of the images of the dataset
                    (height,width, channels) as a tuple
                    Note that this does not include the 'batch' as a dimension.
                    If you have a batch like 'X_train'
                    then you can provide the input_shape using X_train.shape[1:]
    """

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)

    # Zero_Padding:pads the border of X_input with zeros
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    print(X.shape)
    X = Activation('relu')(X)

    # MAX POOL
    X = MaxPool2D((2, 2), name='max_pool')(X)

    # FLATTEN X + FULLY CONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


happyModel = HappyModel(X_train.shape[1:])
comp = happyModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
result_train = happyModel.fit(x=X_train, y=Y_train, batch_size=64, epochs=40)
preds = happyModel.evaluate(x=X_test, y=Y_test)
print()
print('Loss = ' + str(preds[0]))
print('Test Accuracy = ' + str(preds[1]))



