import h5py

import numpy as np

from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras import backend as K

import theano

from skimage.io import imread

def load_VGG_16(weights_path='weights/vgg16_weights.h5'):

    img_width = 250
    img_height = 250
    img = imread('images/Recipe_Images/6698_0.jpg')
    # this will contain our generated image
    input_img = K.placeholder((1, 3, img_width, img_height))

    # build the VGG16 network with our input_img as input
    first_layer = ZeroPadding2D((1, 1), dim_ordering='tf', input_shape=(img_width, img_height, 3))
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
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    return model

def get_activations(model, layer, X_batch):
        '''
        INPUT:  (1) Keras Sequential model object
                (2) integer: The layer to extract weights from
                (3) 4D numpy array: All the X data you wish to extract
                    activations for
        OUTPUT: (1) numpy array: Activations for that layer
        '''
        input_layer = model.layers[0].input
        specified_layer_output = model.layers[layer].get_output(train=False)
        theano_activation_fn = theano.function([input_layer],
                                        specified_layer_output,
                                        allow_input_downcast=True)
        activations = theano_activation_fn(X_batch)
        return activations

if __name__ == '__main__':

    img = imread('images/Recipe_Images/6698_0.jpg')
    img2 = imread('images/Recipe_Images/6698_1.jpg')
    imgs = [img, img2]
    X_batch = np.array(imgs)

    model = load_VGG_16()



    # get_activations = theano.function([model.layers[0].input], model.layers[28].output(train=False), allow_input_downcast=True)

    activations = get_activations(model, 29, X_batch)
