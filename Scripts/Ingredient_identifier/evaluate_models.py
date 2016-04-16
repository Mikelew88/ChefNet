'''
This module contains code used for testing various modes, and predicting ingredients in an image for anechdotal testing
'''

import cPickle as pickle
import sys
import numpy as np
import pandas as pd
import json

from sklearn.metrics import log_loss, classification_report, confusion_matrix, f1_score

from skimage.io import imread
from skimage.transform import resize

from keras.models import model_from_json

from vectorize_data import vectorize_text, load_imgs
from preprocess_df import create_df_image_key

sys.path.append('../Preprocessing')
from load_VGG_net import load_VGG_16, get_activations

def load_model_and_vocab(model_name):
    ''' Load pickled model and list of words '''

    model = model_from_json(open('/data/models/'+model_name+'_architecture.json').read())
    model.load_weights('/data/models/'+model_name+'_weights.h5')

    with open('/data/vocab/small_vocab.pkl', 'r') as fp:
        vocab = pickle.load(fp)

    return model, vocab

def write_img_caption(model, vocab, img_arr, df, img_id=None, threshold=.5):
    ''' Predict words in an img '''

    preds = model.predict(img_arr, verbose=0)[0]

    if img_id:
        print 'Recipe Number: ' + str(img_id)

    pred_index = np.where(preds > threshold)[0]

    generated = ''

    for i, next_index in enumerate(pred_index):

        next_word = vocab[next_index]

        if generated == '':
            generated += next_word
            sys.stdout.write(next_word)
            sys.stdout.write(' (' + str(preds[next_index]) + ')')

        else:
            generated += ', '+next_word
            sys.stdout.write(', ')
            sys.stdout.write(next_word)
            sys.stdout.write(' (' + str(preds[next_index]) + ')')

        sys.stdout.flush()

    if img_id:
        print '\n'

        id_int = int(img_id)
        true_y = df.query('id == @id_int')
        print 'The dish is called {}'.format(true_y['item_name'].values)
        print 'These are the true ingredients: '
        for item in true_y['ingred_list']:
            print item
    pass

def load_jpg(img_path, img_shape, vgg=True):
    ''' Convert .jpgs to array compatible with model and predict '''

    img_x, img_y = img_shape
    img = imread(img_path)
    img = resize(img, (img_x, img_y,3))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    if vgg:
        model = load_VGG_16(img_x, '/data/weights/vgg16_weights.h5')

        img_array = np.empty((1, 3, img_x, img_y))
        img_array[0,:,:,:] = img
        final_array = get_activations(model, 30, img_array)
    else:
        final_array = np.expand_dims(img, axis=1)

    return final_array

def load_preprocessed(img_id, img_num, img_folder):
    ''' Prepare data for captioning that has been preprocessed '''

    img_arr = np.load('/data/'+img_folder+str(img_id)+'_'+str(img_num)+'.npy')

    if img_folder == 'preprocessed_imgs/':
        img_arr = np.expand_dims(img_arr, axis=0)

    return img_arr, img_id

def make_cutoff(y, threshold):
    ''' helper fuction to convert predicted probabilites to boolean '''

    y_pred = np.zeros((y.shape), dtype=bool)
    for row, obs in enumerate(y):
        for col, val in enumerate(obs):
            if val > threshold:
                y_pred[row,col] = True
            else:
                y_pred[row,col] = False
    return y_pred

def validation_metrics(model, vocab, input_shape, img_path, threshold = .5):
    ''' Generate some metrics with validation set '''

    with open('/data/dfs/test_df.pkl', 'r') as f:
        test_df = pickle.load(f)

    test_df_expanded = create_df_image_key(test_df, img_path)

    X_test = load_imgs(np.array(test_df_expanded['img_path']), input_shape)
    y_true = vectorize_text(np.array(test_df_expanded['clean_ingred']), vocab)

    y_pred = model.predict(X_test)
    ''' find best threshold to maximize f1 score '''
    # for i,food in enumerate(vocab):
    #     print food+':'
    #     for t in np.arange(.1, .9, .1):
    #         y_pred_cats = make_cutoff(y_pred, t)
    #         print 'Threshold {}: {}'.format(t, np.mean(f1_score(y_true[:,i], y_pred_cats[:,i])))

    y_pred_cats = make_cutoff(y_pred, .25)

    print classification_report(y_true,y_pred_cats, target_names=list(vocab))

    print 'Average Log Loss Score: {}'.format(np.mean(log_loss(y_true,y_pred)))



if __name__ == '__main__':
    df = pd.read_csv('/data/dfs/recipe_data.csv')
    # Other model: MLP_full_batch
    model, vocab = load_model_and_vocab('VGG_full')

    # Other folder: /data/preprocessed_imgs/
    # (512,3,3)
    validation_metrics(model, vocab, (512,3,3), '/data/vgg_imgs/', .25)
    # img_array, img_id = load_preprocessed(8452, 4, 'vgg_imgs/')
    # write_img_caption(model, vocab, img_array, df, img_id = img_id, threshold=.3)
    #
    # print '\n'
    #
    # img_array = load_jpg('/data/Recipe_Images/237315_0.jpg', (100,100))
    # write_img_caption(model, vocab, img_array, df, threshold=.3)
