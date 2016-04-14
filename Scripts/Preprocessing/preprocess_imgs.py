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

from train_VGG_net import load_VGG_16, get_activations

''' Image Processing '''

def load_imgs(img_arrays, input_shape):
    x, y, z = input_shape
    X = np.empty((len(img_arrays),x,y,z))
    for i, img in enumerate(img_arrays):
        X[i,:,:,:] = np.load(img)

    return X

def preprocess_imgs(base_path, img_keys):
    ''' Save .jpgs arrays and VGG net decomposed arrays

    Input:
        Series of image paths

    Output:
        Array of vectorized images, and list of rows to drop due to bad images
    '''
    img_size = 100
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
    ''' Run on scraped images to create numpy arrays and VGG net processed data
    '''
    df = pd.read_csv(base_path+'recipe_data.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # train_df, test_df = create_validation_set(df)

    id_key, df_expanded = create_df_image_key(df, base_path+'Recipe_Images/')

    preprocess_imgs(base_path, df_expanded['file_key'])
    pass

if __name__ == '__main__':
    pass
