import sys

sys.dont_write_bytecode = True

import os
import re
import pandas as pd
import numpy as np
import operator
import itertools
import indicoio

from collections import defaultdict

from itertools import izip_longest

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from skimage.io import imread_collection, imread
from skimage.transform import resize

from load_VGG_net import load_VGG_16, get_activations

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

''' Image Processing '''

def load_imgs(img_arrays, input_shape):
    x, y, z = input_shape
    X = np.empty((len(img_arrays),x,y,z))
    for i, img in enumerate(img_arrays):
        X[i,:,:,:] = np.load(img)

    return X

def preprocess_imgs(img_keys):
    ''' Save .jpgs arrays and VGG net decomposed arrays

    Input:  series of image paths

    Output: array of vectorized images, and list of rows to drop due to bad images
    '''
    img_size = 100
    base_path='/data/'
    model = load_VGG_16(img_size, base_path+'weights/vgg16_weights.h5')

    for img_key in img_keys:
        img = imread(base_path+'Recipe_Images/'+img_key+'.jpg')
        img = resize(img, (img_size,img_size,3))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        np.save(base_path+'preprocessed_imgs/'+img_key+'.npy', img)
        print 'Saved image: {}'.format(img_key)

        img_array = np.empty((1, 3, img_size, img_size))
        img_array[0,:,:,:] = img
        activation = get_activations(model, 30, img_array)

        np.save(base_path+'vgg_imgs/'+img_key+'.npy', activation)
        print 'Save VGG array: {}'.format(img_key)
    pass

def save_processed_imgs_to_disk(base_path='/data/'):
    ''' Run on scraped images to create numpy arrays and VGG net processed data '''
    df = pd.read_csv(base_path+'recipe_data.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # train_df, test_df = create_validation_set(df)

    id_key, df_expanded = create_df_image_key(df, base_path+'Recipe_Images/')

    preprocess_imgs(base_path, df_expanded['file_key'])
    pass

def find_unprocessed_imgs():
    ''' Helper function to start preprecessing wherever stopped '''

    processed_dir = [x.split('.')[0] for x in os.listdir('/data/preprocessed_imgs/')]
    unprocessed_dir = [x.split('.')[0] for x in os.listdir('/data/Recipe_Images/')]

    return list(set(unprocessed_dir)-set(processed_dir))


if __name__ == '__main__':
    unprocessed = find_unprocessed_imgs()
    preprocess_imgs(unprocessed)
