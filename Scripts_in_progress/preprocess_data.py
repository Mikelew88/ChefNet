import sys

sys.dont_write_bytecode = True

import os
import re
import pandas as pd
import numpy as np
import itertools

from itertools import izip_longest
from itertools import izip_longest

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from keras.preprocessing.image import ImageDataGenerator

from skimage.io import imread_collection, imread
from skimage.transform import resize

from train_VGG_net import load_VGG_16, get_activations


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

''' Text processing '''

def clean_text(ingred_list):
    ''' Clean ingredient text to only keep key words for Vectorizer

    Input:
        List of ingredients as scraped

    Output:
        Underscored string for each ingredient
    '''

    wordnet_lemmatizer = WordNetLemmatizer()

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
                    , 'pint', 'pints', 'fluid', 'cold', 'about', 'circles' \
                    , 'your', 'favorite', 'room', 'temperature', 'skinless' \
                    , 'blanched', 'beaten', 'thawed', 'lightly', 'light' \
                    , 'fourth', 'at', 'broken', 'quart', 'freshly', 'drain' \
                    , 'reserve', 'liquid', 'degree', 'mashed', 'square' \
                    , 'on', 'crosswise', 'strip', 'with', 'tail', 'attached' \
                    , 'coating', 'according', 'direction', 'end']
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
                word = wordnet_lemmatizer.lemmatize(word)
                if word not in exclude_words:
                    line_str.append(word)
            if line_str != []:
                line_final.append(line_str)
        ingred_caption.append(line_final)

    ingred_caption_final = []
    for row in ingred_caption:
        # row_final=['#START#']
        row_final = []
        for item in row:
            item_final = ' '.join(item)
            item_final = item_final.strip('-')

            row_final.append(str(item_final))

        # row_final.append('#END#')

        ingred_caption_final.append(row_final)

    # ingred_for_vectorizer = [', '.join(x) for x in ingred_caption_underscored]

    return ingred_caption_final

def create_text_vectorizer(text_list):
    ''' Convert Ingredients to Count Vectors

    Input:
        Raw ingredient list as scraped

    Output:
        Count vector
    '''

    vocab = sorted(set(itertools.chain.from_iterable(text_list)))

    corpus = []
    for recipe in text_list:
        for word in recipe:
            corpus.append(word)

    # text = ' '.join(text_list)

    print 'corpus length: ' + str(len(corpus))

    print 'total words: ' + str(len(vocab))
    word_indices = dict((c, i) for i, c in enumerate(vocab))
    indices_word = dict((i, c) for i, c in enumerate(vocab))

    return word_indices, indices_word

def vectorize_text(text_list, word_indices):
    ''' Vectorize multiple cleaned lists of ingredients '''
    y = np.zeros((len(text_list), len(word_indices)), dtype=np.bool)
    for i, recipe in enumerate(text_list):
        for t, word in enumerate(recipe):
            y[i, word_indices[word]] = 1
    return y

def tensorize_text(text_list, word_indices, max_caption_len):
    y = np.zeros((len(text_list), len(word_indices)), dtype=np.bool)
    X = np.zeros((len(sentences), max_caption_len, len(chars)), dtype=np.bool)

    for i, recipe in enumerate(text_list):
        for t, word in enumerate(recipe):
            X[i, t, word_indices[word]] = 1
        y[i, t, word_indices[text_list[i+1]]] = 1
    return X, y

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
    base_path = '../'
    df = pd.read_csv(base_path+'data/recipe_data.csv')
    # vectorizer, words = vectorize_text(df['ingred_list'], 1000)
    text_list = clean_text(df['ingred_list'])
    vect = create_text_vectorizer(text_list)
    binarizer = MultiLabelBinarizer()
    vocab = set(itertools.chain.from_iterable(text_list))
    binarizer = binarizer.fit(vocab)
    # y, indices_word = vectorize_text(df['clean_text'])
    # X, y = tensorize_text(words[:1])
    # array, words = vectorize_text(test, 10000)