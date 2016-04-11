''' Base code from Keras

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import random
import sys

from preprocess_data import clean_text

import sys

sys.dont_write_bytecode = True


df = pd.read_csv('../data/recipe_data.csv')

words, vocab = clean_text(df['ingred_list'])

corpus = []
for recipe in words:
    for word in recipe:
        corpus.append(word)

text_list = words
# text = ' '.join(text_list)

print('corpus length:', len(corpus))

print('total words:', len(vocab))
word_indices = dict((c, i) for i, c in enumerate(sorted(vocab)))
indices_word = dict((i, c) for i, c in enumerate(sorted(vocab)))

print('Vectorization...')
y = np.zeros((len(text_list), len(vocab)), dtype=np.bool)
for i, sentence in enumerate(text_list):
    for t, word in enumerate(sentence):
        y[i, word_indices[word]] = 1

import pdb; pdb.set_trace()

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(vocab))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text_list) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = text_list[start_index: start_index + maxlen]
        generated.append(sentence)
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(vocab)))
            for t, words in enumerate(sentence):
                x[0, t, word_indices[word]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            if next_word == '#':
                break
            generated.append(next_word)
            sentence = sentence[1:] + next_word

            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()
