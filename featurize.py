import pandas as pd
import numpy as np

from spacy.en import English

from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import GRU
from keras.layers.embeddings import Embedding
from keras.layers.core import TimeDistributedDense, RepeatVector, Merge, Flatten, Dense, Dropout
from keras.models import Sequential

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD

import codecs

from preprocess_data import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer


max_caption_len = 16
vocab_size = 10000



def get_ingredient_tensor_timeseries(ingredient_lst, nlp, timesteps):
    '''
    Returns a time series of word vectors for tokens in the ingredients

    Input:
    ingredient_lst: list of unicode objects
    nlp: an instance of the class English() from spacy.en
    timesteps: the number of

    Output:
    A numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)
    '''
    nb_samples = len(ingredient_lst)
    word_vec_dim = nlp(ingredient_lst[0])[0].vector.shape[0]
    ingred_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in xrange(len(ingredient_lst)):
        tokens = nlp(ingredient_lst[i])
    	for j in xrange(len(tokens)):
    		if j<timesteps:
    			ingred_tensor[i,j,:] = tokens[j].vector

    return ingred_tensor

def get_questions_matrix_sum(ingredient_lst, nlp):
	'''
	Sums the word vectors of all the tokens in a question

	Input:
	questions: list of unicode objects
	nlp: an instance of the class English() from spacy.en

	Output:
	A numpy array of shape: (nb_samples, word_vec_dim)
	'''
	assert not isinstance(questions, basestring)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_matrix = np.zeros((nb_samples, word_vec_dim))
	for i in xrange(len(questions)):
		tokens = nlp(questions[i])
		for j in xrange(len(tokens)):
			questions_matrix[i,:] += tokens[j].vector

	return questions_matrix

if __name__ == '__main__':
    from pymongo import MongoClient
    #Store results in mongo
    db_client = MongoClient()
    db = db_client['allrecipes']
    recipe_db = db['recipe_data']

    df = pd.DataFrame(list(recipe_db.find()))

    ingred_list = df['ingred_list']

    clean_ingreds = clean_text(ingred_list)
