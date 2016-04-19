''' This module vectorizes imgs and text for models. It also contains code that created a smaller vocabulary which only contains words that are seen more than 100 times in the data '''

import sys

sys.dont_write_bytecode = True

import os
import re
import pandas as pd
import numpy as np
import operator
import itertools
import indicoio
import matplotlib
# This allows me to plot on AWS
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import cPickle as pickle

from collections import defaultdict

from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud


''' Image Processing '''

def load_imgs(img_arrays, input_shape):
    x, y, z = input_shape
    X = np.empty((len(img_arrays),x,y,z))
    for i, img in enumerate(img_arrays):
        X[i,:,:,:] = np.load(img)

    return X

''' Text processing '''

def clean_text(ingred_list):
    ''' Clean ingredient text to only keep key words for Vectorizer

    Input:
        List of ingredients as scraped

    Output:
        string off all ingredients to search over
    '''

    wordnet_lemmatizer = WordNetLemmatizer()

    ingred_caption = []
    # iterate over recipes
    for item in ingred_list:
        line_final = []
        # iterate over recipe ingredient line items (in mongo db these did not need to be split)
        for line in item.replace(' or ', ', ').split(','):
            line_str = []
            for word in line.split():
                word = word.lower().strip()
                word = word.strip('-')
                word = word.strip('[]')
                word = ''.join(e for e in word if e.isalnum() and not e.isdigit())
                word = wordnet_lemmatizer.lemmatize(word)
                line_str.append(word)

            if line_str != []:
                line_final.append(line_str)
        ingred_caption.append(line_final)

    ingreds_clean = []
    for row in ingred_caption:
        row_final = []
        for item in row:
            item_final = ' '.join(item)
            item_final = item_final.strip('-')

            row_final.append(str(item_final))

        row_final = ' | '.join(row_final)
        ingreds_clean.append(row_final)

    return ingreds_clean

def vectorize_text(clean_text, vocab):
    ''' Vectorize multiple cleaned lists of ingredients '''

    y = np.zeros((len(clean_text), len(vocab)), dtype=np.bool)
    for i, text in enumerate(clean_text):
        for t, voc in enumerate(vocab):
            if voc in text:
                y[i, t] = 1
    return y

def create_small_vocab(y, vocab):
    ''' Create a subset of the vocabulary with words that apear more than 100 times '''

    ingred_counts = np.sum(y, axis=0)

    small_vocab = []

    for i, vocab in zip (ingred_counts, vocab):
        if i > 100 and vocab:
            small_vocab.append(vocab)

    return small_vocab


def save_vocab(vocab, name):
    with open('/data/'+name+'.pkl', 'w') as f:
        pickle.dump(vocab, f)


def generate_wordcloud(y, vocab):
    ''' Generate a simple word cloud of text '''
    ingred_counts = np.sum(y, axis=0)

    word_cloud_text = []

    for i, vocab in zip(ingred_counts, vocab):
        word_cloud_text.append((str(vocab),int(i)))

    # Generate a word cloud image
    wordcloud = WordCloud(background_color = "white")

    wordcloud.fit_words(word_cloud_text)

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('../../figures/vocab_wordcloud.png')
    pass


if __name__ == '__main__':

    # with open('/data/vocab/small_vocab.pkl', 'r') as fp:
    #     vocab = pickle.load(fp)
    #
    # df = pd.read_csv('/data/dfs/recipe_data.csv')
    # # vectorizer, words = vectorize_text(df['ingred_list'], 1000)
    # text_list = clean_text(df['ingred_list'])
    # y = vectorize_text(text_list, vocab)
    # #
    # # small_vocab, word_cloud_text = create_small_vocab(y, vocab)
    # #
    # # save_vocab(small_vocab, 'small_vocab')
    #
    # # generate_wordcloud(y, vocab)
    pass
