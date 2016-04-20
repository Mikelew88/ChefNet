'''
This module contains code used for testing various modes, and predicting ingredients in an image for anechdotal testing
'''

import cPickle as pickle
import sys
import numpy as np
import pandas as pd
import json

import matplotlib
# This allows me to plot on AWS
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.metrics import log_loss, classification_report, precision_recall_fscore_support, f1_score, accuracy_score

from skimage.io import imread
from skimage.transform import resize

from keras.models import model_from_json

from vectorize_data import vectorize_text, load_imgs
from preprocess_df import create_df_image_key

from build_models import build_VGG_net

sys.path.append('../Preprocessing')
from load_VGG_net import load_VGG_16, get_activations

def load_model_and_vocab(model_name):
    ''' Load pickled model and list of words '''
    base_dir = '/data/'

    model = model_from_json(open(base_dir+'models/'+model_name+'_architecture.json').read())
    model.load_weights(base_dir+'models/'+model_name+'_weights.h5')

    with open(base_dir+'vocab/small_vocab.pkl', 'r') as fp:
        vocab = pickle.load(fp)

    return model, vocab

def write_img_caption(model, vocab, img_arr, df=None, img_id=None, threshold=.5):
    ''' Predict words in an img '''

    preds = model.predict(img_arr, verbose=0)[0]
    # preds =

    if img_id:
        print 'Recipe Number: ' + str(img_id)

    pred_index = np.argsort(preds)[::-1]
    print '______________________'
    print
    print 'ChefNet Predictions: '
    print

    generated = ''

    for i, next_index in enumerate(pred_index):

        next_word = vocab[next_index]

        if preds[next_index]<threshold:
            break

        if generated == '':
            generated += next_word
            sys.stdout.write(next_word)
            sys.stdout.write(' (' + str(int(round(preds[next_index]*100))) + '%)')

        else:
            generated += ', '+next_word
            sys.stdout.write(', ')
            sys.stdout.write(next_word)
            sys.stdout.write(' (' + str(int(round(preds[next_index]*100))) + '%)')

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

def load_jpg(img_path, vgg=True):
    ''' Convert .jpgs to array compatible with model and predict '''

    img_x, img_y = (100,100)
    img = imread(img_path)
    img = resize(img, (img_x, img_y,3))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    if vgg:
        model = load_VGG_16(img_x, '../../vgg_weights/vgg16_weights.h5')

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

    y_pred_cats = make_cutoff(y_pred, threshold)

    print classification_report(y_true,y_pred_cats, target_names=list(vocab))

    # print 'Average Log Loss Score: {}'.format(np.mean(log_loss(y_true,y_pred)))

    Correct = y_pred_cats == y_true

    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_pred_cats)

    print 'Mean Recall: '
    print np.mean(recall)
    print
    print 'Mean Precision: '
    print np.mean(precision)


    top_items(recall, 'recall')
    print
    top_items(precision, 'precision')

    return y_true, y_pred_cats, precision, recall, fbeta_score, support

def top_items(metric, name):
    print 'Top ten categories for {}:'.format(name)

    for i in np.argsort(metric)[-10:]:
        print '{}: {}'.format(metric[i], vocab[i])

def generate_wordcloud(vocab, metric, name):
    ''' Generate a simple word cloud of text '''
    list_tuples = []
    for w, c in zip(vocab, metric):
        list_tuples.append((w,int(c*100)))

    # Generate a word cloud image
    wordcloud = WordCloud(background_color = "white")

    wordcloud.fit_words(list_tuples)

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('../../figures/'+name)
    pass

def random_simulation(y_true):
    ''' generate a random simulation to determine baseline recall and precision '''

    y_rand = np.random.rand(y_true.shape[0], y_true.shape[1])

    y_rand_cats = make_cutoff(y_rand,.937)
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_true, y_rand_cats)
    print
    print 'Random Guessing...'
    print 'Mean Recall: '
    print np.mean(recall)
    print
    print 'Mean Precision: '
    print np.mean(precision)
    print

    print classification_report(y_true,y_rand_cats, target_names=list(vocab))


    return y_rand_cats

# 0.37      0.41
if __name__ == '__main__':
    df = pd.read_csv('/data/dfs/recipe_data.csv')
    # Other model: MLP_full_batch
    model, vocab = load_model_and_vocab('VGG_3_dropout')

    # Other folder: /data/preprocessed_imgs/
    # (3,100,100)
    y_true, y_pred_cats, precision, recall, fbeta_score, support = validation_metrics(model, vocab, (512,3,3), '/data/vgg_imgs/', .25)
    #
    # y_rand_cats = random_simulation(y_true)

    # generate_wordcloud(vocab,recall, 'recall_wordcloud.png')
    # random_guessing = calculate_random_guessing(y, vocab)

    #
    # img_array, img_id = load_preprocessed(8694, 6, 'vgg_imgs/')
    # write_img_caption(model, vocab, img_array, df, img_id = img_id, threshold=.25)
    #
    # print '\n'
    #
    pass
