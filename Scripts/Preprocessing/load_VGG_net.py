import h5py
import sys

import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras import backend as K

import theano

from skimage.io import imread

sys.dont_write_bytecode = True

def load_VGG_16(img_size, weights_path='../weights/vgg16_weights.h5'):

    img_width = img_size
    img_height = img_size
    # this will contain our generated image
    input_img = K.placeholder((1, 3, img_width, img_height))

    # build the VGG16 network with our input_img as input
    first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))
    first_layer.input = input_img

    model = Sequential()
    model.add(first_layer)
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        print k
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    return model

def get_activations(model, layer, X_batch):
    '''
    Save second to last VGG activations for a batch of images
    INPUT:
        model = Keras Sequential model object
        layer = integer of the layer to extract weights from
        X_batch = 4D numpy array of all the X data you wish to extract activations for
    OUTPUT:
        numpy array: Activations for that layer
    '''
    input_layer = model.layers[0].input
    specified_layer_output = model.layers[layer].get_output(train=False)
    theano_activation_fn = theano.function([input_layer],
                                    specified_layer_output,
                                    allow_input_downcast=True)
    activations = theano_activation_fn(X_batch)

    return activations

if __name__ == '__main__':
    pass
    # img = imread('../images/Recipe_Images/6698_0.jpg')
    # # img2 = imread('../images/Recipe_Images/6698_1.jpg')
    #
    # X_batch = np.empty((1, 3, 250, 250))
    # X_batch[0,:,:,:]=np.swapaxes(img, 0, 2)
    #
    # model = load_VGG_16()
    #
    # # (X, 512, 7, 7)
    # activations = get_activations(model, 30, X_batch)
