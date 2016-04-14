import sys

sys.dont_write_bytecode = True
sys.setrecursionlimit(999999999)

import json

import pandas as pd
import numpy as np
from itertools import izip_longest

from sklearn.metrics import log_loss

import cPickle as pickle

from preprocess_text_el import clean_text_basic, vectorize_text

from build_models import build_MLP_net, build_VGG_net, build_LSTM_net

def batch_train(train_df, test_df, model, input_shape, word_indices, indices_word, word_keyword, epochs = 1, batch_size = 50):
    ''' Since all images do not fit into memory, we must batch process ourselves
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

                accuracy, loss = model.train_on_batch(X_train, y_train, accuracy=True)

                print 'Batch {} \n Accuracy: {} \n Loss: {}'.format(i, accuracy, loss)

        X_test = load_imgs(np.array(test_df['img_path']), input_shape)
        y_test = vectorize_text(np.array(test_df['clean_ingred']), vocab)

        y_pred = model.predict(X_test)
        print 'Epoch {}'.format(e)
        print 'Mean Validation Log Loss: {}'.format(np.mean(log_loss(y_test,y_pred)))
        print '\n \n'

    return model, word_indices

def grouper(iterable, n, fillvalue=None):
    ''' helper function for batching '''
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

def pickle_trained_nn(model, name):
    ''' Save Pickle of trained net '''
    with open('/data/models/'+name+'.pkl', 'w') as f:
        pickle.dump(model, f)
    pass

def train_net(model_function=build_VGG_net, save_name = 'VGG_sigmoid_el'):
    ''' Train and save a VGG preprocessed net '''
    # max_classes=len(vocab)
    img_path = '/data/temp_imgs_bigger/vgg_imgs/'
    input_shape = (512,3,3)

    df = pd.read_csv('/data/recipe_data.csv')

    df['clean_ingred'] = clean_text_basic(df['ingred_list'])

    with open('../data/el_keywords.pkl', 'r') as fp:
        vocab = pickle.load(fp)

    train_df, test_df = create_validation_set(df)

    train_df_expanded = create_df_image_key(train_df, img_path)
    test_df_expanded = create_df_image_key(test_df, img_path)

    train_df, test_df = create_validation_set(df)

    base_path = '/data/'
    model = model_function(len(vocab), input_shape)
    trained_model = batch_train(train_df_expanded, test_df_expanded, model, input_shape, vocab, epochs=10)

    pickle_trained_nn(model, save_name)

    with open('/data/models/words_'+save_name+'.pkl', 'wb') as f:
        pickle.dump(indices_word, f)

    with open('/data/models/word_keyword_'+save_name+'.pkl', 'wb') as f:
        pickle.dump(word_keyword, f)

    return trained_model, indices_word

if __name__ == '__main__':
    # trained_model, words = train_net(model_function=build_LSTM_net, save_name = 'LSTM_sigmoid_bigger')
    trained_model, words = train_net()
