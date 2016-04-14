from bs4 import BeautifulSoup

import urllib

from nltk.stem import WordNetLemmatizer
import cPickle as pickle


def scrape_text(url):
    ''' Quick scraper to grab potential list of ingredients '''

    """define opener"""
    class MyOpener(urllib.FancyURLopener):
        version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
    myopener = MyOpener()

    page = myopener.open(url)
    soup = BeautifulSoup(page, 'lxml')

    tables = soup.find_all('tr', {'align': 'center', 'valign':'top'})[:24]
    wordnet_lemmatizer = WordNetLemmatizer()
    exclude_list = ['appetizer', 'appetite', 'ate', 'bake', 'baked alaska' \
    , 'bland', 'boil', 'bowl', 'breakfast', 'brunch', 'burrito', 'calorie' \
    , 'carbohydrate', 'cater', 'chef', 'chew', 'chow', 'comestible' \
    , 'cook', 'crisp', 'cuisine', 'cupcake', 'diet', 'digest' \
    , 'digestive system', 'dine', 'diner', 'dinner', 'dip', 'dish', 'dried' \
    , 'dry', 'eat', 'edible', 'fast', 'fat', 'feast', 'fed', 'feed', 'fire' \
    , 'food', 'food pyramid', 'foodstuff', 'fork', 'freezer', 'fired' \
    , 'fry', 'gastronomy', 'glasses', 'grated', 'grub', 'hunger', 'hungry' \
    , 'jug', 'julienne', 'junk food', 'kettle', 'kitchen', 'knife', 'ladle' \
    , 'loaf', 'lunch', 'lunch box', 'main course', 'menu', 'micronutrient' \
    , 'broil', 'comestible', 'cookbook', 'cupboard', 'foodstuff', 'mug' \
    , 'nibble', 'nosh', 'nourish', 'nourishment', 'nutrition', 'nutritious' \
    , 'omlet', 'omnivore', 'order', 'oven', 'pan', 'pate', 'patty', 'picnic' \
    , 'pitcher', 'plate', 'platter', 'poached', 'pot', 'provision', 'punch' \
    , 'ration', 'recipe', 'refreshment', 'refrigerator', 'restaurant', 'roll' \
    , 'rolling pin', 'sandwich', 'slice', 'smoked', 'snack', 'soup', 'spicy' \
    , 'spoon', 'spork', 'stomach', 'stove', 'straw', 'stringy' \
    , 'sub sandwich', 'submarine sandwich', 'supper', 'sustenance', 'sweet' \
    , 'take-out', 'tart', 'teapot', 'toaster', 'torte', 'tuber', 'vegetable' \
    , 'vitamin', 'wok', 'hot','pop', 'cake', 'pop', 'sour', 'tea', 'ice']
    food_labels = []
    for i in tables:
        text = i.get_text()
        text = text.split('\n')
        for j in text:
            text = str(j).lower().strip()

            if text != '' and text != '\r' and text:
                text_lemm = str(wordnet_lemmatizer.lemmatize(text))
                if text_lemm not in exclude_list:
                    food_labels.append(text_lemm)

    vocab = sorted(set(food_labels))

    # small bandaid to add the one word that i missed, forgot to scrape z
    vocab.append('zucchini')
    return sorted(vocab)

if __name__ == '__main__':

    food_labels = scrape_text('http://www.enchantedlearning.com/wordlist/food.shtml')

    with open('/data/el_vocab_4_14.pkl', 'w') as f:
        pickle.dump(food_labels, f)
