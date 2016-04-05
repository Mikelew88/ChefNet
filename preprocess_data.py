from os import listdir
import re
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

def vectorize_imgs(ingred_list, dir_path='images/Recipe_Images'):
    datagen = ImageDataGenerator()
    img_dir = listdir(dir_path)

    # We must copy the ingredient vector for each distinct image for a single recipe
    img_paths = [dir_path+x for x in img_dir if x.split('_')[0] in id_list]

    for i in temp:
        print i
    return imread_collection(img_paths, conserve_memory=True)

if __name__ == '__main__':
    temp = vectorize_imgs(['6698', '6788'])
