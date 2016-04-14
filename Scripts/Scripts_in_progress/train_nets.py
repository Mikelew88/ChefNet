import sys

sys.dont_write_bytecode = True
sys.setrecursionlimit(999999999)

import json

import pandas as pd
import numpy as np
from itertools import izip_longest

from sklearn.metrics import log_loss

import cPickle as pickle

from vectorize_text import clean_text, vectorize_text
from prepare_data_for_model import create_validation_set, create_df_image_key, load_imgs

from build_models import build_MLP_net, build_VGG_net, build_LSTM_net

def batch_train(train_df, test_df, model, input_shape, vocab, epochs = 10, batch_size = 50, print_loss=False):
    ''' If all images do not fit into memory, we must batch process ourselves
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

    return model, vocab

def grouper(iterable, n, fillvalue=None):
    ''' helper function for batching '''
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def save_nn(model, name):
    # save as JSON
    json_string = model.to_json()
    open('/data/models/'+name+'_architecture.json', 'w').write(json_string)

    model.save_weights('/data/models/'+name+'_weights.h5')
    pass

def train_net(model_function=build_VGG_net, save_name = 'test', img_path='/data/temp_imgs_bigger/vgg_imgs/', input_shape=(512,3,3), fits_in_memory = True):
    ''' Train and save NN '''

    df = pd.read_csv('/data/recipe_data.csv')

    df['clean_ingred'] = clean_text(df['ingred_list'])

    with open('/data/small_vocab.pkl', 'r') as fp:
        vocab = pickle.load(fp)

    train_df, test_df = create_validation_set(df)

    train_df_expanded = create_df_image_key(train_df, img_path)
    test_df_expanded = create_df_image_key(test_df, img_path)

    model = model_function(len(vocab), input_shape)

    if fits_in_memory:
        print 'Loading data... '
        X_test = load_imgs(test_df_expanded['img_path'], input_shape)
        y_test = vectorize_text(test_df_expanded['clean_ingred'], vocab)
        model.fit(X_train, y_train, nb_epoch=10, validation_data = (X_test, y_test))
    else:
        trained_model = batch_train(train_df_expanded, test_df_expanded, model, input_shape, vocab, epochs=10, batch_size=64)

    save_nn(model, save_name)
    print save_name + ' has been saved'

    return model, vocab

if __name__ == '__main__':
    trained_model, words = train_net(model_function=build_MLP_net, save_name = 'MLP_full_batch', img_path = '/data/preprocessed_imgs/', input_shape = (3,100,100), fits_in_memory=False)
    # trained_model, vocab = train_net(save_name = 'VGG_full', img_path = '/data/vgg_imgs/')
