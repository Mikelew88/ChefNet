import sys

sys.dont_write_bytecode = True

import os
import re
import pandas as pd
import numpy as np
import operator
import itertools

import cPickle as pickle

from vectorize_data import clean_text

def create_validation_set(df):
    ''' Create set of recipes that will be set aside for validation '''
    np.random.seed(seed=33)
    msk = np.random.rand(len(df), ) < 0.90
    train_df = df[msk]
    test_df = df[~msk]

    # np.save('/data/msk.npy', msk)

    return train_df, test_df

def create_df_image_key(df_in, dir_path):
    ''' Join ingredient lists to image locations, one to many

    Input:  (1) dataframe of unique recipes
            (2) local path to image folder

    Output: (1) dataframe with one row for each image of a recipe
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

def Save_Train_and_Test_df():
    ''' save dfs to disk for training and testing '''

    df = pd.read_csv('/data/dfs/recipe_data.csv')

    df['clean_ingred'] = clean_text(df['ingred_list'])

    train_df, test_df = create_validation_set(df)

    with open('/data/dfs/train_df.pkl', 'w') as f:
        pickle.dump(train_df, f)

    with open('/data/dfs/test_df.pkl', 'w') as f:
        pickle.dump(test_df, f)
    pass



if __name__ == '__main__':
    Save_Train_and_Test_df()
