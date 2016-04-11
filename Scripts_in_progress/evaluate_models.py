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

if __name__ == '__main__':
    df = pd.read_csv('/data/recipe_data.csv')

    model, words = open_pkl_and_words('MLP_VGG')
    pred_words, true_y = predict_img(model, words, '8694', '4', 'vgg_imgs/', df)
