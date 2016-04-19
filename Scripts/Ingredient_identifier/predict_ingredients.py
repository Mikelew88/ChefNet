import numpy as np
import cPickle as pickle

from evaluate_models import load_jpg, write_img_caption

from build_models import build_VGG_net

def load_model_and_vocab_local():
    ''' Load pickled model and list of words '''

    base_dir = '../../'

    with open(base_dir+'models/small_vocab.pkl', 'r') as fp:
        vocab = pickle.load(fp)

    model = build_VGG_net(len(vocab), (512,3,3))
    model.load_weights(base_dir+'models/TL_3_dropout_weights.h5')

    return model, vocab

def predict_user_photo(model, vocab):
    file_name = raw_input("Please enter the image file name: ")

    try:
        img_array = load_jpg('../../images/'+file_name)
        write_img_caption(model, vocab, img_array, threshold=.25)
    except:
        print 'Please enter a valid file name, make sure to include filetype, .jpg for example'
        predict_user_photo(model, vocab)
        break
    pass

if __name__ == '__main__':
    model, vocab = load_model_and_vocab_local()
    predict_user_photo(model, vocab)
