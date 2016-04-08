import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

from preprocess_data import create_validation_set, create_df_image_key, load_imgs, vectorize_text

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def batch_train(df, model, max_classes, epochs = 10, batch_size = 50, img_path='/data/preprocessed_imgs'):
    '''
    Since all images do not fit into memory, we must batch process ourselves
    '''
    df['y'], words = vectorize_text(df['ingred_list'], max_classes)

    for i in epochs:
        # Shuffle df rows
        df.reindex(np.random.permutation(df.index))

        train_df, test_df = create_validation_set(df)

        train_df_expanded = create_df_image_key(train_df, img_path)
        test_df_expanded = create_df_image_key(test_df, img_path)


        for i, df_batch in enumerate(grouper(train_df_expanded, batch_size)):

            y_train = train_df_expanded['y']
            X_train = load_imgs(train_df_expanded['img_path'])

            model.train_on_batch(X_train, y_train,accuracy=True)

            print accuracy

        X_test = load_imgs(test_df_expanded['img_path'])
        y_test = test_df_expanded['y']

        y_pred = model.predict_proba(X_test)
        print 'Epoch {}'.format(i)
        print 'Mean R2 Score: {}'.format(np.mean(r2_score(y_test,y_pred)))

        return model, words


def build_MLP_net(max_classes, img_size=100):
    '''
    Create a preliminary Keras model
    '''

    nb_classes = max_classes
    # images are RGB
    img_channels = 3

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_size, img_size)))
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
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


if __name__ == '__main__':
    max_classes=10000
    base_path = '/data/'
    df = pd.read_csv(base_path+'recipe_data.csv')
    model = build_MLP_net(max_classes)
    trained_model, words = batch_train(df, model, max_classes,  img_path='/data/temp_imgs/preprocessed_imgs')
