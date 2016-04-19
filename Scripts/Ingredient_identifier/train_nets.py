import sys

sys.dont_write_bytecode = True
sys.setrecursionlimit(999999999)

import json

import pandas as pd
import numpy as np
from itertools import izip_longest

from sklearn.metrics import log_loss

import cPickle as pickle

from vectorize_data import clean_text, vectorize_text, load_imgs
from preprocess_df import create_validation_set, create_df_image_key

from build_models import build_MLP_net, build_VGG_net, build_LSTM_net, build_random

def batch_train(train_df, test_df, model, input_shape, vocab, epochs = 10, batch_size = 50, print_loss=False):
    ''' Batch processing for when images don't all fit in memory

    Input:  (1) traing dataframe
            (2) validation dataframe
            (3) keras model to be trained
            (4) shape of image data, (3,100,100) or (512,3,3)
            (5) vocabulary of words, classes being predicted
            (6) epochs to trian on
            (7) batch size of epics
            (8) if true, will print loss and accuracy of each batch

    Output: (1) trained keras model

    '''

    for e in range(1,epochs+1):
        # Shuffle df rows for each epoch
        train_df = train_df.iloc[np.random.permutation(len(train_df))]

        train_X = np.array(train_df['img_path'])
        train_y = np.array(train_df['clean_ingred'])
        train_array = np.vstack((train_X, train_y)).T

        # skip last batch
        stop_batch = train_array.shape[0]/batch_size

        for i, batch in enumerate(grouper(train_array, batch_size)):
            if i < stop_batch:
                batch = np.array(batch)
                X_train = load_imgs(batch[:,0], input_shape)

                y_train = vectorize_text(batch[:,1], vocab)
                test_0 = np.where(y_train[0,:]==True)[0]
                test_where = np.where(y_train == True)[1]

                if print_loss:
                    accuracy, loss = model.train_on_batch(X_train, y_train, accuracy=True)
                    print 'Batch {} \n Accuracy: {} \n Loss: {}'.format(i, accuracy, loss)
                else:
                    model.train_on_batch(X_train, y_train)

        X_test = load_imgs(np.array(test_df['img_path']), input_shape)
        y_test = vectorize_text(np.array(test_df['clean_ingred']), vocab)

        y_pred = model.predict(X_test)
        print 'Epoch {}'.format(e)
        print 'Mean Validation Log Loss: {}'.format(np.mean(log_loss(y_test,y_pred)))
        print '\n \n'

    return model

def grouper(iterable, n, fillvalue=None):
    ''' helper function for batch selection '''

    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def save_nn(model, name):
    ''' Save neural net structure as json and weights as h5 '''

    json_string = model.to_json()
    open('/data/models/'+name+'_architecture.json', 'w').write(json_string)

    model.save_weights('/data/models/'+name+'_weights.h5')
    pass

def train_net(model_function=build_VGG_net, save_name = 'test', img_path='/data/temp_imgs_bigger/vgg_imgs/', input_shape=(512,3,3), fits_in_memory = True):
    ''' Train and save NN

    Input:  (1) model arcitecture from build models
            (2) file name to save model under
            (3) image path of preporcessed image files, vgg_imgs or preprocessed_imgs
            (4) shape of image data, (512,3,3) for vgg or (3,100,100) for preprocessed
            (5) if false, model will be trained in batch to allow for training on more data than fits in memory

    Output: (1) trained keras model

    '''

    with open('/data/vocab/small_vocab.pkl', 'r') as fp:
        vocab = pickle.load(fp)

    with open('/data/dfs/train_df.pkl', 'r') as f:
        train_df = pickle.load(f)

    with open('/data/dfs/test_df.pkl', 'r') as f:
        test_df = pickle.load(f)

    train_df_expanded = create_df_image_key(train_df, img_path)
    test_df_expanded = create_df_image_key(test_df, img_path)

    model = model_function(len(vocab), input_shape)

    if fits_in_memory:
        print 'Loading data... '
        X_train = load_imgs(train_df_expanded['img_path'], input_shape)
        y_train = vectorize_text(train_df_expanded['clean_ingred'], vocab)

        X_test = load_imgs(test_df_expanded['img_path'], input_shape)
        y_test = vectorize_text(test_df_expanded['clean_ingred'], vocab)
        model.fit(X_train, y_train, nb_epoch=25, validation_data = (X_test, y_test))
    else:
        trained_model = batch_train(train_df_expanded, test_df_expanded, model, input_shape, vocab, epochs=25, batch_size=64)

    save_nn(model, save_name)
    print save_name + ' has been saved'

    return model

def random_simulation():
    ''' Randomly assign labels to see how random guessing would perform '''

    with open('/data/vocab/small_vocab.pkl', 'r') as fp:
        vocab = pickle.load(fp)

    with open('/data/dfs/train_df.pkl', 'r') as f:
        train_df = pickle.load(f)

    X_train = train_df['id']
    y_train = vectorize_text(train_df['clean_ingred'], vocab)

    model = build_random(len(vocab))
    model.fit(X_train, y_train)
    save_nn(model, 'random_simulation')
    return model


if __name__ == '__main__':
    # trained_model = train_net(model_function=build_MLP_net, save_name = 'dumb_net', img_path = '/data/preprocessed_imgs/', input_shape = (3,100,100), fits_in_memory=False)
    # trained_model = train_net(save_name = 'VGG_3_dropout', img_path = '/data/vgg_imgs/')

    random = random_simulation()
