import cPickle as pickle
import sys
import numpy as np
import pandas as pd
import json

from preprocess_data import clean_text

def open_pkl_and_words(model_name):
    ''' Load pickled model and list of words '''
    with open('/data/models/'+model_name+'.pkl', 'r') as f:
        model = pickle.load(f)

    with open('/data/models/words_'+model_name+'.pkl', 'rb') as fp:
        indices_word = pickle.load(fp)

    # with open('/data/models/words_'+model_name+'.txt', 'r') as f:
    # words = json.loads('/data/models/words_'+model_name+'.txt')
    return model, indices_word
    # , words

def predict_img(model, words, id, img_num, img_folder, df):
    ''' Allow user to test model with a specific image'''
    img_arr = np.load('/data/'+img_folder+id+'_'+img_num+'.npy')

    pred = np.array(model.predict(img_arr))
    pred_words = words[pred[0,:] > 0.5]

    print 'Net thinks there are these ingredinets: '
    for item in pred_words:
        print item
    print ''

    id_int = int(id)
    true_y = df.query('id == @id_int')
    print 'The dish is called {}'.format(true_y['item_name'].values)
    print 'These are the true ingredients: '
    for item in true_y['ingred_list'].values:
        print item

    return pred_words, true_y

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argsort(np.random.multinomial(1, a, 1))[::-1]

def write_img_caption(model, indices_word, id, img_num, img_folder, df):

    img_arr = np.load('/data/'+img_folder+id+'_'+img_num+'.npy')
    preds = model.predict(img_arr, verbose=0)[0]

    # for diversity in [0.2, 0.5, 1.0, 1.2]:
    print 'Recipe Number: ' + id
    # print '----- diversity: ' + str(diversity)

    generated = ''
    print '----- Generating with Img: ' + img_num
    sys.stdout.write(generated)

    # sorted_index = np.argsort(preds)[::-1]

    pred_index = np.where(preds > .5)[0]

    for i, next_index in enumerate(pred_index):

        next_word = indices_word[next_index]

        if next_word == '#':
            next_word = ''

        # if i > 10:
        #     break
        if generated == '':
            generated += next_word
            sys.stdout.write(next_word)
        else:
            generated += ', '+next_word
            sys.stdout.write(', ')
            sys.stdout.write(next_word)
        # if next_word == '#':
        #     break
        # if i == 0:
        #     sys.stdout.write(next_word)
        sys.stdout.flush()

    print '\n'

    id_int = int(id)
    true_y = df.query('id == @id_int')
    print 'The dish is called {}'.format(true_y['item_name'].values)
    print 'These are the true ingredients: '
    for item in clean_text(true_y['ingred_list']):
        print item

    return preds


if __name__ == '__main__':
    df = pd.read_csv('/data/recipe_data.csv')

    model, indices_word = open_pkl_and_words('VGG_sigmoid_bigger')
    pred_words = write_img_caption(model, indices_word, '6788', '0', 'vgg_imgs/', df)
