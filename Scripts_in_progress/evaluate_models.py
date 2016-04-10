import cPickle as pickle
import numpy as np

def open_pkl_and_words(model_name):
    with open('/data/models/'+model_name+'_temp.pkl', 'r') as f:
        model = pickle.load(f)
    words = np.load('/data/models/words_'+model_name)
    return model, words


if __name__ == '__main__':
    model, words = open_pkl_and_words(MLP_VGG)
