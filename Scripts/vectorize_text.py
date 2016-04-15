import sys

sys.dont_write_bytecode = True

import os
import re
import pandas as pd
import numpy as np
import operator
import itertools
import indicoio

import cPickle as pickle

from collections import defaultdict

from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import MultiLabelBinarizer

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
    ingred_counts = np.sum(y, axis=0)

    small_vocab = []
    word_cloud_text = ''

    for i, vocab in zip (ingred_counts, vocab):
        if i > 100 and vocab:
            small_vocab.append(vocab)
            word_cloud_text.join(' '+vocab*i)

    return small_vocab, word_cloud_text


def save_vocab(vocab, name):
    with open('/data/'+name+'.pkl', 'w') as f:
        pickle.dump(vocab, f)



def generate_wordcloud(word_cloud_text):
    ''' Generate a simple word cloud of vocabulary '''
    from wordcloud import WordCloud



    # Generate a word cloud image
    wordcloud = WordCloud().generate(word_cloud_text)

    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    plt.imshow(wordcloud)
    plt.axis("off")

    # take relative word frequencies into account, lower max_font_size
    wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(word_cloud_text)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':

    with open('/data/el_vocab_4_14.pkl', 'r') as fp:
        vocab = pickle.load(fp)

    df = pd.read_csv('/data/recipe_data.csv')
    # vectorizer, words = vectorize_text(df['ingred_list'], 1000)
    text_list = clean_text_basic(df['ingred_list'])
    y = vectorize_text(text_list, vocab)

    small_vocab, word_cloud_text = create_small_vocab(y, vocab)

    save_vocab(small_vocab, 'small_vocab')
