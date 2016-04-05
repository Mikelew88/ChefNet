from os import listdir
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread_collection, imshow_collection

def clean_text(ingred_list):
    '''
    Vectorize ingredient text into TFIDF

    Input:
    Output:
    '''
    ingred_caps = []
    exclude_chars = '[/1234567890().-:,]'
    exclude_words = ['teaspoons', 'teaspoon', 'tablespoon', 'tablespoons' \
                     , 'cup', 'cups', 'ounce', 'ounces', 'bunch', 'bunches' \
                     , 'large', 'fluid ounce','fluid ounces', 'can', 'cans' \
                    , 'pound', 'pounds', 'dash', 'dry', 'lean', 'jars', 'to' \
                    , 'taste', 'slice', 'slices', 'clove', 'cloves' \
                    , 'cube', 'cubes', 'bag', 'bags', 'package', 'packages' \
                    , 'inch', 'inches', 'for', 'a', 'recipe', 'peeled' \
                    , 'grated', 'chopped', 'optional', 'prepared', 'finely' \
                    , 'crushed', 'degrees', 'F', 'C', 'bottle', 'bottles' \
                    , 'rinsed', 'sliced', 'softened', 'halves', 'halved' \
                     ,'cubed', 'drained', 'optional', 'ground', 'or' , '-' \
                    , 'pounded', 'thick', 'diced', 'pinch', 'minced', 'box' \
                    , 'boxes', 'cleaned', 'and', 'cut', 'into', 'rings' \
                    , 'frozen', 'shredded']

    ingred_caption = []
    for item in ingred_list:
        line_final = []
        for line in item:
            line_str = []
            line = re.sub(exclude_chars, '', line)
            for word in line.split():
                if word not in exclude_words:
                    line_str.append(word)
            line_final.append(line_str)
        ingred_caption.append(line_final)

    ingred_caption_underscored = [['_'.join(x) for x in y] for y in ingred_caption]

    return ingred_caption_underscored

def expand_df_images(df_in, dir_path='images/Recipe_Images'):
    datagen = ImageDataGenerator()
    img_dir = listdir(dir_path)

    # We must copy the ingredient vector for each distinct image for a single recipe
    dir_index = [x.split('_')[0] for x in img_dir]
    img_path = [dir_path+x for x in img_dir]

    df_dir = pd.DataFrame(img_path, index=dir_index)
    df_dir.columns = ['img_path']

    df_in.index = df_in['id']

    df_out = df_in.merge(df_dir, how='left', left_index=True, right_index=True)
    return df_out

def vectorize_imgs(img_paths):

    return imread_collection(img_paths, conserve_memory=True)

def vectorize_text(ingred_list):
    underscored_captions = clean_text(ingred_list)

    ingred_for_vectorizer = [' '.join(x) for x in underscored_captions]

    vectorizer=TfidfVectorizer()
    trans_vect =vectorizer.fit_transform(ingred_for_vectorizer)
    array = trans_vect.toarray()
    words = vectorizer.get_feature_names()
    return array, words

if __name__ == '__main__':
    temp = vectorize_imgs(['6698', '6788'], ['eggs', 'potato'])
