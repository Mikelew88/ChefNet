'''
This module contains various neural network Architecture that I played around with. All net construction comes from here.
'''

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

def build_MLP_net(nb_classes, input_shape):
    ''' This acrcutecture was suggested by Keras for the NSIT handrwitten number database, I found it did well, but not as well as when I leveraged VGG net for help '''

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    # train the model using SGD + momentum
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=Adam())

    print 'We have a model!'

    return model

def build_VGG_net(nb_classes, input_shape):
    ''' This arcutecture proved the most fruitful, utilizing transfer learning from VGG net '''

    model = Sequential()

    model.add(Convolution2D(512, 3, 3, border_mode='same',
                        input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4608))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(4608))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=Adam())

    print 'We have a model!'

    return model

def build_LSTM_net(nb_classes, input_shape):
    ''' This net was used to try and pass each convolution from VGG as a tensor for and LSTM, it was very slow to trian and didn't prove very fruitful '''

    model = Sequential()

    model.add(Convolution2D(512, 3, 3, border_mode='same',
                        input_shape=input_shape))
    model.add(Reshape((512, 9)))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=Adam())

    print 'We have a LSTM model!'

    return model

def build_RNN(nb_classes, input_shape, max_caption_len):
    ''' This was an artifact of trying to take an image captioning approach to learn ingredients, I felt the code was interesting, but the approach was not useful for my goal of guessing the components of a dish '''


    img_model = Sequential()

    img_model.add(Convolution2D(512, 3, 3, border_mode='same',
                        input_shape=input_shape))
    img_model.add(Flatten())
    img_model.add(Dense(512))

    language_model = Sequential()
    language_model.add(Embedding(nb_classes, 256, input_length=max_caption_len))
    language_model.add(LSTM(output_dim=128, return_sequences=True))
    language_model.add(TimeDistributedDense(128))

    image_model.add(RepeatVector(max_caption_len))

    model = Sequential()

    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    # let's encode this vector sequence into a single vector
    model.add(LSTM(256, return_sequences=False))
    # which will be used to compute a probability
    # distribution over what the next word in the caption should be!
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def build_random(nb_classes):

        model = Sequential()

        model.add(Dense(1))
        model.add(Dense(nb_classes))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=Adam())
        return model
