import os
import re
import pandas as pd
import numpy as np

from itertools import izip_longest
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread_collection
from skimage.transform import resize
from itertools import izip_longest
from train_VGG_net import load_VGG_16, get_activations
# from scipy.misc import imread

import sys

sys.dont_write_bytecode = True

def create_validation_set(df):
    '''
    Create set of recipes that will be set aside for validation
    '''
    np.random.seed(seed=33)
    msk = np.random.rand(len(df), ) < 0.90
    train_df = df[msk]
    test_df = df[~msk]

    np.save('/data/msk.npy', msk)

    return train_df, test_df

def create_df_image_key(df_in, dir_path):
    '''
    Join ingredient lists to image locations, one to many

    Input:
        df_in = dataframe of unique recipes
        dir_path = local path to image folder

    Output:
        DataFrame with one row for each image of a recipe
    '''

    # We must copy the ingredient vector for each distinct image for a single recipe
    # datagen = ImageDataGenerator()
    img_dir = os.listdir(dir_path)

    dir_index = [int(x.split('_')[0]) for x in img_dir]

    img_paths = [dir_path+x for x in img_dir]

    df_dir = pd.DataFrame(img_paths, index=dir_index)
    df_dir.columns = ['img_path']

    df_in.index = df_in['id']

    df_out = df_in.merge(df_dir, how='inner', left_index=True, right_index=True)

    id_key = np.array(df_out['id'])

    np.save('/data/id_key.npy', id_key)

    return id_key, df_out

def clean_text(ingred_list):
    '''
    Clean ingredient text to only keep key words for TFIDF

    Input:
        List of ingredients as scraped

    Output:
        Underscored string for each ingredient
    '''
    ingred_caps = []
    exclude_chars = '[/1234567890().-:,]'
    exclude_words = ['teaspoons', 'teaspoon', 'tablespoon', 'tablespoons' \
                     , 'cup', 'cups', 'ounce', 'ounces', 'bunch', 'bunches' \
                     , 'large', 'fluid ounce','fluid ounces', 'can', 'cans' \
                    , 'pound', 'pounds', 'dash', 'dry', 'lean', 'jars', 'to' \
                    , 'taste', 'slice', 'slices', 'clove', 'cloves' \
                    , 'cube', 'cubes', 'bag', 'bags', 'package', 'packages' \
                    , 'inch', 'inches', 'for', 'a', 'recipe', 'peeled' \
                    , 'grated', 'chopped', 'optional', 'prepared', 'finely' \
                    , 'crushed', 'degrees', 'F', 'C', 'bottle', 'bottles' \
                    , 'rinsed', 'sliced', 'softened', 'halves', 'halved' \
                     ,'cubed', 'drained', 'optional', 'ground', 'or' , '-' \
                    , 'pounded', 'thick', 'diced', 'pinch', 'minced', 'box' \
                    , 'boxes', 'cleaned', 'and', 'cut', 'into', 'rings' \
                    , 'frozen', 'shredded']

    ingred_caption = []
    for item in ingred_list:
        line_final = []
        for line in item:
            line_str = []
            line = re.sub(exclude_chars, '', line)
            for word in line.split():
                if word not in exclude_words:
                    line_str.append(word)
            line_final.append(line_str)
        ingred_caption.append(line_final)

    ingred_caption_underscored = [['_'.join(x) for x in y] for y in ingred_caption]

    return ingred_caption_underscored

def vectorize_imgs(img_paths):
    '''
    Convert .jpgs to arrays, resized and swap channel axis

    Input:
        Series of image paths

    Output:
        Array of vectorized images, and list of rows to drop due to bad images
    '''
    model = load_VGG_16()

    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return izip_longest(*args, fillvalue=fillvalue)

    for i, img_batch in enumerate(grouper(img_paths, 10000)):

        img_gen = imread_collection(img_batch, conserve_memory=True)
        img_array = np.empty((len(img_batch),3,250,250))

        for img in img_gen:
            img = np.swapaxes(img, 0, 2)
            img_array[i,:,:,:] = np.swapaxes(img, 1, 2)

        activation = get_activations(model, 30, img_array)
        np.save('/data/Image_Arrays/array_'+str(i)+'.npy', img_array)
        np.save('/data/VGG_Arrays/VGG_array_'+str(i)+'.npy', activations)
    pass

def vectorize_text(ingred_list, max_classes=10000):
    '''
    Convert Ingredients to Count Vectors

    Input:
        Raw ingredient list as scraped

    Output:
        Count vector
    '''

    underscored_captions = clean_text(ingred_list)

    ingred_for_vectorizer = [' '.join(x) for x in underscored_captions]

    vectorizer=CountVectorizer(max_features=max_classes)

    trans_vect =vectorizer.fit_transform(ingred_for_vectorizer)
    array = trans_vect.toarray()
    words = vectorizer.get_feature_names()
    return array, words

if __name__ == '__main__':
    #preprocess all images
    # dir_path = 'images/Recipe_Images'
    # img_dir = os.listdir(dir_path)
    #
    # dir_index = [int(x.split('_')[0]) for x in img_dir]

    df = pd.read_csv('/data/recipe_data.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)

    train_df, test_df = create_validation_set(df)

    id_key, df_expanded = create_df_image_key(train_df, '/data/Recipe_Images/')

    vectorize_imgs(df_expanded['img_path'].values)
