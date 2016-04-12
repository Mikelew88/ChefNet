import cPickle as pickle
import numpy as np
import pandas as pd

def open_pkl_and_words(model_name):
    ''' Load pickled model and list of words '''
    with open('/data/models/'+model_name+'_temp.pkl', 'r') as f:
        model = pickle.load(f)
    words = np.load('/data/models/words_'+model_name+'.npy')
    return model, words

def predict_img(model, words, id, img_num, img_folder, df):
    ''' Allow user to test model with a specific image'''
    img_arr = np.load('/data/'+img_folder+id+'_'+img_num+'.npy')
    pred = np.array(model.predict(img_arr))
    pred_words = words[pred[0,:] > 0.01]

    print 'Net thinks there are these ingredinets: '
    for item in pred_words:
        print item
    print ''

    id_int = int(id)
    true_y = df.query('id == @id_int')
    print 'The dish is called {}'.format(true_y['item_name'].values)
    print 'These are the true ingredients: '
    for item in true_y['ingred_list'].values:
        print item

    return pred_words, true_y

def write_img_caption(img_num):
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print ''
        print '----- diversity: ' + str(diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + img_num + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(sentence):
                x[0, t, word_indices[word]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            if next_word == '#':
                break

            generated += next_word
            sentence = sentence[1:] + next_word

            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()


if __name__ == '__main__':
    df = pd.read_csv('/data/recipe_data.csv')

    model, words = open_pkl_and_words('LSTM')
    pred_words, true_y = predict_img(model, words, '8694', '4', 'preprocessed_imgs/', df)
