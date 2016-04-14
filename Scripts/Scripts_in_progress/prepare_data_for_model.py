import sys

sys.dont_write_bytecode = True

import os
import re
import pandas as pd
import numpy as np
import operator
import itertools

def create_validation_set(df):
    ''' Create set of recipes that will be set aside for validation
    '''
    np.random.seed(seed=33)
    msk = np.random.rand(len(df), ) < 0.90
    train_df = df[msk]
    test_df = df[~msk]

    # np.save('/data/msk.npy', msk)

    return train_df, test_df

def create_df_image_key(df_in, dir_path):
    ''' Join ingredient lists to image locations, one to many

    Input:
        df_in = dataframe of unique recipes
        dir_path = local path to image folder

    Output:
        DataFrame with one row for each image of a recipe
    '''

    # We must copy the ingredient vector for each distinct image for a single recipe
    img_dir = os.listdir(dir_path)

    dir_index = [int(x.split('_')[0]) for x in img_dir]

    img_paths = [dir_path+x for x in img_dir]

    df_dir = pd.DataFrame(img_paths, index=dir_index)
    df_dir.columns = ['img_path']

    df_in.index = df_in['id']

    df_out = df_in.merge(df_dir, how='inner', left_index=True, right_index=True)

    df_out['file_key'] = [x.split('/')[-1].split('.')[0] for x in df_out['img_path']]

    return df_out

''' Image Processing '''

def load_imgs(img_arrays, input_shape):
    x, y, z = input_shape
    X = np.empty((len(img_arrays),x,y,z))
    for i, img in enumerate(img_arrays):
        X[i,:,:,:] = np.load(img)

    return X

if __name__ == '__main__':
    base_path = '../'
    df = pd.read_csv(base_path+'data/recipe_data.csv')
    # vectorizer, words = vectorize_text(df['ingred_list'], 1000)
    text_list = clean_text(df['ingred_list'])

    word_indices, indices_word, word_keyword = create_text_vectorizer(text_list)
    # vocab = sorted(set(itertools.chain.from_iterable(text_list)))
    # indicoio_keywords = indicoio.keywords(vocab, version=2)
    # ingred_caption_keywords = []
    # for i in indicoio_keywords:
    #     try:
    #         keyword = str(max(i.iteritems(), key=operator.itemgetter(1))[0])
    #         ingred_caption_keywords.append(keyword)
    #     except:
    #         pass
    #
    # new_vocab = sorted(set(ingred_caption_keywords))
    # print new_vocab
    # print len(new_vocab)

    # y, indices_word = vectorize_text(df['clean_text'])
    # X, y = tensorize_text(words[:1])
    # array, words = vectorize_text(test, 10000)
