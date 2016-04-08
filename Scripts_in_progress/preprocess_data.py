import os
import re
import pandas as pd
import numpy as np

from itertools import izip_longest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread_collection, imread
from skimage.transform import resize
from itertools import izip_longest
from train_VGG_net import load_VGG_16, get_activations
# from scipy.misc import imread

from unidecode import unidecode

import sys

sys.dont_write_bytecode = True

# from sklearn.cross_validation import train_test_split

# from pymongo import MongoClient

def prepare_data(df, img_path = 'images/Recipe_Images/'):
    ''' Prepare Images and Ingredients for NN

    Input:
        df = Scraped data

    Output:
        Training and test vector representations of Image and Ingredient data
    '''
    msk = np.random.rand(len(df)) < 0.9
    train_df = df[msk]
    test_df = df[~msk]

    train_df = expand_df_images(train_df, img_path)
    test_df = expand_df_images(test_df, img_path)
    import pdb; pdb.set_trace()

    X_train, y_train = vectorize_data(train_df, text_classes)
    X_test, y_test =  vectorize_data(test_df, text_classes)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train, y_train, X_test, y_test

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
    # datagen = ImageDataGenerator()
    img_dir = os.listdir(dir_path)

    dir_index = [int(x.split('_')[0]) for x in img_dir]

    img_paths = [dir_path+x for x in img_dir]

    df_dir = pd.DataFrame(img_paths, index=dir_index)
    df_dir.columns = ['img_path']

    df_in.index = df_in['id']

    df_out = df_in.merge(df_dir, how='inner', left_index=True, right_index=True)

    df_out['file_key'] = [x.split('/')[-1].split('.')[0] for x in df_out['img_path']]

    return df_out

def clean_text(ingred_list):
    ''' Clean ingredient text to only keep key words for Vectorizer

    Input:
        List of ingredients as scraped

    Output:
        Underscored string for each ingredient
    '''
    ingred_caps = []
    exclude_words = ['teaspoons', 'teaspoon', 'tablespoon', 'tablespoons' \
                     , 'cup', 'cups', 'ounce', 'ounces', 'bunch', 'bunches' \
                     , 'large', 'medium', 'small', 'in', 'half', 'lengthwise' \
                    , 'pound', 'pounds', 'dash', 'dry', 'lean', 'jars', 'to' \
                    , 'taste', 'slice', 'slices', 'clove', 'cloves' \
                    , 'cube', 'cubes', 'bag', 'bags', 'package', 'packages' \
                    , 'inch', 'inches', 'for', 'a', 'recipe', 'peeled' \
                    , 'grated', 'chopped', 'optional', 'prepared', 'finely' \
                    , 'crushed', 'degrees', 'f', 'c', 'bottle', 'bottles' \
                    , 'rinsed', 'sliced', 'softened', 'halves', 'halved' \
                     ,'cubed', 'drained', 'optional', 'ground', 'or' , '-' \
                    , 'pounded', 'thick', 'diced', 'pinch', 'minced', 'box' \
                    , 'boxes', 'cleaned', 'and', 'cut', 'into', 'rings' \
                    , 'frozen', 'shredded', 'trimmed', 'fresh', 'taste' \
                    , '', ' ', 'uncooked', 'raw', 'bulk', 'pieces', 'piece' \
                    , 'drop', 'drops', 'can', 'cans', 'fluid ounce' \
                    , 'fluid ounces', 'boneless', 'boned', 'bone' \
                    , 'containers', 'container', 'cook', 'cooked', 'cooking' \
                    , 'unhusked', 'unpeeled', 'trays', 'tub', 'tubs' \
                    , 'zested', 'of', 'one', 'very', 'thin', 'thinly', 'on' \
                    , 'all', 'naural', 'organic', 'farm', 'raised', 'fresh' \
                    , 'pint', 'pints', 'fluid', 'cold', 'about', 'circles']
    ingred_caption = []

    # iterate over recipes
    for item in ingred_list:
        line_final = []
        # iterate over recipe ingredient line items (in mongo db these did not need to be split)
        for line in item.split(','):
            line_str = []
            for word in line.split():
                word = word.lower().strip()
                word = word.strip('-')
                word = word.strip('[]')
                word = ''.join(e for e in word if e.isalnum() and not e.isdigit())
                if word not in exclude_words:
                    line_str.append(word)
            if line_str != []:
                line_final.append(line_str)
        ingred_caption.append(line_final)

    exclude_final = ['', ' ', 'room_temperature', 'skinless', 'bone', 'in' \
    , 'blue', 'blanched', 'black', 'all', 'tri', 'your_favorite', \
    'zested_juiced', 'your_choice_of_shape', 'your_choice', 'warm', 'divided']

    ingred_caption_underscored = []
    for row in ingred_caption:
        row_final=[]
        for item in row:
            item_final = '_'.join(item)
            item_final = item_final.strip('-')
            item_final = item_final.strip('[]')
            item_final = item_final.strip('_')


            if item_final not in exclude_final:
                row_final.append(item_final)
        ingred_caption_underscored.append(row_final)

    ingred_for_vectorizer = [' '.join(x) for x in ingred_caption_underscored]

    return ingred_for_vectorizer


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

def vectorize_text(clean_text, max_classes):
    ''' Convert Ingredients to Count Vectors

    Input:
        Raw ingredient list as scraped

    Output:
        Count vector
    '''

    vectorizer=CountVectorizer(max_features=max_classes)
    vectorizer.fit(clean_text)
    # trans_vect = vectorizer.transform(ingred_for_vectorizer)

    # array = trans_vect.toarray()
    words = vectorizer.get_feature_names()
    return vectorizer, words

def load_imgs(img_arrays, img_size):
    X = np.empty((len(img_arrays),3,img_size, img_size))

    for i, img in enumerate(img_arrays):
        X[i,:,:,:] = np.load(img)

    return X

def save_processed_imgs_to_disk(base_path='/data/'):
    '''
    Run on scraped images to create numpy arrays and VGG net processed data
    '''
    df = pd.read_csv(base_path+'recipe_data.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # train_df, test_df = create_validation_set(df)

    id_key, df_expanded = create_df_image_key(df, base_path+'Recipe_Images/')

    preprocess_imgs(base_path, df_expanded['file_key'])

if __name__ == '__main__':
    base_path = '../'
    df = pd.read_csv(base_path+'data/recipe_data.csv')
    vectorizer, words = vectorize_text(df['ingred_list'], 5000)
    # array, words = vectorize_text(test, 10000)
