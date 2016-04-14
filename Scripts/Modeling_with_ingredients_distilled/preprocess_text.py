''' Here is a script I wrote to distill ingredients out of the raw scraped text. I eneded up going with a predetermined list of ingredients, but nonetheless felt like this was interesting.'''

import sys

sys.dont_write_bytecode = True

import os
import re
import pandas as pd
import numpy as np
import operator
import itertools
import indicoio

from collections import defaultdict

from itertools import izip_longest

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from skimage.io import imread_collection, imread
from skimage.transform import resize

from train_VGG_net import load_VGG_16, get_activations

indicoio.config.api_key = os.environ['G_INDICO_API_KEY']

def clean_text(ingred_list, max_ingred_len=3):
    ''' Clean ingredient text to only keep key words for Vectorizer

    Input:
        List of ingredients as scraped

    Output:
        Underscored string for each ingredient
    '''

    wordnet_lemmatizer = WordNetLemmatizer()

    ''' Here begins my long list of words to exclude '''
    exclude_words = ['teaspoons', 'teaspoon', 'tablespoon', 'tablespoons' \
                     , 'cup', 'cups', 'ounce', 'ounces', 'bunch', 'bunches' \
                     , 'large', 'medium', 'small', 'in', 'half', 'lengthwise' \
                    , 'pound', 'pounds', 'dash', 'dry', 'lean', 'jars', 'to' \
                    , 'taste', 'slice', 'slices', 'clove', 'cloves' \
                    , 'cube', 'cubes', 'bag', 'bags', 'package', 'packages' \
                    , 'inch', 'inches', 'for', 'a', 'recipe', 'peeled' \
                    , 'grated', 'chopped', 'optional', 'prepared', 'finely' \
                    , 'crushed', 'degrees', 'f', 'c', 'bottle', 'bottles' \
                    , 'rinsed', 'sliced', 'softened', 'halves', 'halved' \
                     ,'cubed', 'drained', 'optional', 'ground', 'or' , '-' \
                    , 'pounded', 'thick', 'diced', 'pinch', 'minced', 'box' \
                    , 'boxes', 'cleaned', 'and', 'cut', 'into', 'rings' \
                    , 'frozen', 'shredded', 'trimmed', 'fresh', 'taste' \
                    , '', ' ', 'uncooked', 'raw', 'bulk', 'pieces', 'piece' \
                    , 'drop', 'drops', 'can', 'cans', 'fluid ounce' \
                    , 'fluid ounces', 'boneless', 'boned', 'bone' \
                    , 'containers', 'container', 'cook', 'cooked', 'cooking' \
                    , 'unhusked', 'unpeeled', 'trays', 'tub', 'tubs' \
                    , 'zested', 'of', 'one', 'very', 'thin', 'thinly', 'on' \
                    , 'all', 'naural', 'organic', 'farm', 'raised', 'fresh' \
                    , 'pint', 'pints', 'fluid', 'cold', 'about', 'circles' \
                    , 'your', 'favorite', 'room', 'temperature', 'skinless' \
                    , 'blanched', 'beaten', 'thawed', 'lightly', 'light' \
                    , 'fourth', 'at', 'broken', 'quart', 'freshly', 'drain' \
                    , 'reserve', 'liquid', 'degree', 'mashed', 'square' \
                    , 'on', 'crosswise', 'strip', 'with', 'tail', 'attached' \
                    , 'coating', 'according', 'direction', 'end', 'some' \
                    , 'meat', 'across', 'against', 'the', 'grain', 'amount' \
                    , 'cover', 'leftover', 'desire', 'seeded', 'stemmed' \
                    , 'filet', 'fillet', 'paste', 'link', 'mix', 'see' \
                    , 'footnote', 'link', 'any', 'flavor', 'color', 'apart' \
                    , 'joint', 'any', 'variety', 'accent', 'enhancer', 'such' \
                    , 'fleischmanns', 'shell', 'combination', 'without' \
                    , 'peel', 'each', 'waterpacked', 'from', 'mexico' \
                    , 'bitesize', 'bitesized', 'bitter', 'bit', 'size' \
                    , 'sized', 'chunk', 'floret', 'you', 'grind', 'ground' \
                    , 'whole', 'pepercorn', 'blender', 'blackened', 'fine' \
                    , 'powder', 'center', 'bonein', 'centercut', 'roast' \
                    , 'cap', 'canned', 'regular', 'fat', 'free' \
                    , 'recommended', 'style', 'blackened', 'old', 'bay' \
                    , 'butterlied', 'needed', 'flat', 'more', 'smart' \
                    , 'balance', 'crisco', 'but', 'not', 'soft', 'fit', 'pan' \
                    , 'still', 'burritosize', 'frank', 'browned', 'kitchen' \
                    , 'bouquet', 'note', 'breast', 'bottom', 'layer', 'round' \
                    , 'crust', 'bottled', 'boiling', 'solution', 'newman' \
                    , 'own', 'boiled', 'hour', 'until', 'soft', 'cheap' \
                    , 'bud', 'flavored', 'mixture', 'cheesecloth' \
                    , 'dissolved', 'chunk', 'each', 'flat', 'grilled' \
                    , 'halve', 'medallion', 'thickness', 'tender', 'stirfry' \
                    , 'carcass', 'from', 'an', 'even', 'cutlet', 'eg' \
                    , 'quarter', 'back', 'part', 'note', 'better', 'than' \
                    , 'tender', 'chilled', 'garnish' \
                    , 'choice', 'chop', 'scraped', 'out', 'coarsely', 'jar' \
                    , 'dried', 'instant', 'removed', 'roasted', 'pie' \
                    , 'filling', 'roll', 'roll', 'packed', 'baby', 'french' \
                    , 'refrigerated', 'loaf', 'mixed', 'filling', 'goya' \
                    , 'jigger', 'other', 'stem', 'top', 'extra', 'bar' \
                    , 'ripe', 'quartered', 'whipped', 'gallon' \
                    , 'processed', 'reduced', 'reducedfat', 'low' \
                    , 'split', 'long', 'natural', 'milliliter', 'tip' \
                    , 'vegetarian', 'concentrate', 'melted', 'wedge' \
                    , 'coarse', 'colored', 'firm', 'mini', 'serving' \
                    , 'sheet', 'sheet', 'choice', 'only', 'separated' \
                    , 'shelled', 'sodium', 'flaked', 'fully', 'if', 'new' \
                    , 'kikkoman', 'substitute', 'brewed', 'carton', 'dayold' \
                    , 'discarded', 'covered', 'no', 'pure', 'sharp', 'sifted' \
                    , 'buttery', 'cooled', 'heinz', 'jumbo', 'loin' \
                    , 'pillsbury', 'progresso', 'campbell', 'warm', 'wrapper' \
                    , 'bite', 'base', 'carbonated', 'country', 'ear' \
                    , 'glutenfree', 'heavy', 'kosher', 'miniature' \
                    , 'granule', 'oz', 'quickcooking', 'real' \
                    , 'roughly', 'philadelphia', 'starter', 'swanson' \
                    , 'wide', 'american', 'chex', 'deep', 'fried' \
                    , 'foil', 'aluminium', 'diagonally', 'dipping' \
                    , 'hershey', 'hulled', 'loosely', 'minute' \
                    , 'preferably', 'premium', 'quick', 'reducedsodium' \
                    , 'rolled', 'scrubbed', 'squeezed', 'beverage' \
                    , 'baby', 'concentrated', 'decorating' \
                    , 'dusting', 'flavoring', 'german', 'grit', 'grand' \
                    , 'inchthinck', 'left', 'before', 'pitted', 'deveined' \
                    , 'total', 'intact', 'tip', 'blend', 'mexicanstyle' \
                    , 'over', 'stewed', 'chineese', 'asian', 'picked', 'over' \
                    , 'low', 'head', 'golden', 'toasted', 'brown', 'stray' \
                    , 'washed', 'patty', 'traditional', 'miamistyle' \
                    , 'tart', 'unbaked', 'xinch', 'spray', 'baker', 'joy' \
                    , 'ball', 'deboned', 'bite', 'skinned', 'tied' \
                    , 'if', 'desired', 'divided', 'double', 'readytouse' \
                    , 'pillsbury', 'ragu', 'deep', 'dish', 'pie', 'crust' \
                    , 'pastry', 'firmly', 'packed', 'jif', 'stirred' \
                    , 'extra', 'crunchy', 'creamy', 'jimmy', 'dean' \
                    , 'original', 'hot', 'dog', 'longgrain', 'converted' \
                    , 'matchstick', 'inchthick', 'foam', 'skimmed', 'off' \
                    , 'slivered', 'so', 'sit', 'third', 'discarded', 'tough' \
                    , 'snapped', 'choke', 'scraped', 'out', 'flesh' \
                    , 'spooned', 'rolling', 'tiny', 'squeezed', 'stem', 'rib' \
                    , 'out', 'together', 'el', 'paso', 'gram', 'powdered' \
                    , 'long', 'mccormick', 'portion', 'only', 'pouch', 'pulp' \
                    , 'easy', 'rolling', 'saltfree', 'metal', 'skewer' \
                    , 'wooden', 'soaked', 'water', 'minute', 'kraft', 'skim' \
                    , 'baked', 'slightly', 'flattened', 'warmed', 'slivered' \
                    , 'topping', 'seasoned', 'stewed', 'guinness', 'strong' \
                    , 'blend', 'sifted', 'unbleached', 'use', 'i', 'hunt' \
                    , 'petite', 'do', 'bar', 'scoop', 'wild', 'quick' \
                    , 'packet', 'liter', 'granulated', 'granular' \
                    , 'aspartame', 'eagle', 'brand', 'progresso', 'muir' \
                    , 'glen', 'brand', 'rind', 'removed', 'wedge', 'diameter' \
                    , 'wheel', 'untrimmed', 'flatcut', 'classico', 'crystal' \
                    , 'crusty', 'decoration', 'dish', 'fl', 'sprig', 'leaf' \
                    , 'flatleaf', 'granular', 'nocalorie', 'surcralose' \
                    , 'splenda', 'fullycooked', 'hillshire', 'litl', 'smokies' \
                    , 'polish', 'homemade', 'hungarian', 'wax', 'paper' \
                    , 'including', 'leaf', 'jamaican', 'korean', 'lukewarm' \
                    , 'nondairy', 'nosaltadded', 'orangeflavored', 'patted' \
                    , 'per', 'lb', 'polish', 'prebaked', 'precooked' \
                    , 'process', 'all', 'purpose', 'allpurpose', 'quality' \
                    , 'ranchstyle', 'rotel', 'original', 'asian', 'sambal' \
                    , 'oelek', 'san', 'marzano', 'preferably', 'italian' \
                    , 'sauteed', 'selfrising', 'short', 'length' \
                    , 'koreanstyle', 'savory', 'scoop', 'sauteed', 'skinon' \
                    , 'split', 'solid', 'pack', 'puree', 'spiral', 'tricolor' \
                    , 'tricolored', 'spring', 'roll', 'wrapper', 'available' \
                    , 'asian', 'market', 'loaf', 'stale', 'french', 'steamed' \
                    , 'divided', 'chinese', 'german', 'stone', 'heavy' \
                    , 'string', 'removed', 'summer', 'assorted', 'then' \
                    , 'moon', 'diagonally', 'thickly', 'up', 'removed' \
                    , 'serving', 'giant', 'valley', 'steamer', 'hidden' \
                    , 'vegan', 'big', 'brushing', 'california', 'count' \
                    , 'dice', 'dryroasted', 'drizzling', 'flaky', 'glutinous' \
                    , 'good', 'gread', 'heated', 'i', 'e', 'ie', 'it' \
                    , 'ingredient', 'lite', 'made', 'microwave' \
                    , 'oldfashioned', 'pasteurized', 'pot', 'popped' \
                    , 'readytouse', 'rubbed', 'smuckers', 'three', 'tiny' \
                    , 'tough', 'additional', 'argo', 'kingsford', 'baking' \
                    , 'by', 'wide', 'long', 'heavy', 'duty', 'family', 'g' \
                    , 'condensed', 'compressed', 'instant', 'condensed' \
                    , 'jiffy', 'cored', 'drink', 'coke', 'coin', 'decaf' \
                    , 'is', 'cholesterolfree', 'chiliseasoned', 'alaskan' \
                    , 'angel', 'allvegetable', 'alfredostyle', 'aged' \
                    , 'nonstick', 'nutritional', 'noboil', 'noniodized' \
                    , 'noninstant', 'nonhydrogenated', 'neck', 'necessary' \
                    , 'navy', 'navel', 'napa', 'mycoprotein' \
                    , 'morton', 'multicolored', 'multigrain'\
                    , 'moisture', 'moist', 'mott', 'mexican' \
                    , 'mild', 'mildly', 'mediumdry', 'meaty' \
                    , 'nocook', 'nonfat', 'concord', 'greasing' \
                    , 'hatch', 'glace', 'glaze', 'fruitflavored' \
                    , 'fourcheesefilled', 'freestone', 'freezer' \
                    , 'freezedried', 'fluffy', 'foamy', 'frenched' \
                    , 'frenchfried', 'juicy', 'juiced', 'julienne' \
                    , 'julienned', 'kellog', 'kaiser', 'jarred', 'juiced' \
                    , 'extrawide', 'eye', 'extruded', 'extrasharp' \
                    , 'extract', 'extrafirm', 'extrahot', 'extralarge' \
                    , 'extralean', 'vermont', 'stove', 'strained', 'stiff' \
                    , 'stir', 'aluminum', 'width', 'sleve', 'planter' \
                    , 'silken', 'silver', 'sieve', 'shortgrain' \
                    , 'young', 'zatarains', 'yardlong', 'zest' \
                    , 'wood', 'wrap', 'x', 'williamsburg', 'wholekernel' \
                    , 'sister', 'six', 'sixth', 'skillet', 'skin', 'slab' \
                    , 'seven', 'shank', 'shape', 'secret', 'seedless' \
                    , 'smooth', 'readytoeat', 'readytoserve', 'rectangle' \
                    , 'reducedcalorie', 'reducedsalt', 'refrigerator' \
                    , 'ra', 'puff', 'preserve', 'protein', 'pet', 'pink' \
                    , 'pale', 'pad', 'pacific', 'pace', 'nacho', 'live' \
                    , 'headless', 'heart', 'flake', 'flakeystyle', 'fatfree' \
                    , 'dr', 'creolstyle', 'creamstyle', 'creamystyle', 'cool' \
                    , 'bob', 'peter', 'lugar', 'tube', 'superfine', 'vine' \
                    , 'snow', 'tyson', 'type', 'virginia', 'regularsize' \
                    , 'shake', 'rough', 'rub', 'quickrise', 'rack', 'rim' \
                    , 'ritz', 'saltine', 'rapit', 'request', 'rest' \
                    , 'reynolds', 'picnic', 'pat', 'open', 'niblet' \
                    , 'mediumhot', 'mediumlarge', 'mediumsize' \
                    , 'mediumspice', 'mediumthick', 'mediumgrain', 'maxwell' \
                    , 'matchstickcut', 'meatless', 'london', 'longhorn' \
                    , 'louisianastyle', 'lower', 'lowersodium', 'lowfat' \
                    , 'lowsodium', 'lump', 'lowmoisture', 'kiss', 'heat' \
                    , 'hellmanns', 'husk', 'hug', 'indian', 'individual' \
                    , 'instruction', 'irish', 'joe', 'grinder', 'grill' \
                    , 'haas', 'ha', 'green', 'friendship', 'frysize' \
                    , 'fresno', 'greek', 'greekstyle', 'ghirardelli' \
                    , 'gimme', 'farmer', 'flakystyle', 'flank', 'florida' \
                    , 'fell', 'dinner', 'dinosaur', 'disco', 'dole' \
                    , 'domestic', 'deepfat', 'deepdish', 'decorator' \
                    , 'curlystyle', 'crumble', 'creole', 'cardboard' \
                    , 'caribbean', 'cajun', 'cajunstyle', 'brazil', 'au' \
                    , 'alphabet', 'alternate', 'alum', 'asianstyle' \
                    , 'assemble', 'backbone', 'basket', 'blood', 'xinches' \
                    , 'st', 'sour', 'soul', 'skirt', 'sleeve', 'slender' \
                    , 'silk', 'single', 'snip', 'snack', 'smokie' \
                    , 'rome', 'root', 'rare', 'rock', 'rapid', 'rapidrise' \
                    , 'rainbow', 'pour', 'porter', 'pod', 'basmati', 'batter' \
                    , 'bavarianstyle', 'beau', 'belgian', 'bowl', 'bow' \
                    , 'breakfast', 'breakstone', 'brick', 'spear', 'spread' \
                    , 'solidpack', 'soften', 'snack', 'southwest', 'southern' \
                    , 'southernstyle', 'soyginger', 'splash', 'sport' \
                    , 'spread', 'sponge', 'squeeze', 'ahi', 'al', 'fresco' \
                    , 'amish', 'cesar', 'crisp', 'curd', 'crumb', 'crown' \
                    , 'diet', 'diamond', 'diagonal', 'devil', 'dutchprocess' \
                    , 'earl', 'eat', 'edible', 'eigth', 'emerils', 'equal' \
                    , 'england', 'excess', 'extralong', 'extravirgin' \
                    , 'expeel', 'farmhousestyle', 'fiddlehead', 'fiesta' \
                    , 'fiestastyle', 'fin', 'finger', 'fire', 'food', 'foo' \
                    , 'fork', 'four', 'one', 'two', 'three', 'five', 'six' \
                    , 'seven', 'eight', 'nine', 'fritter', 'freshground' \
                    , 'frenchstyle', 'freshfroxen', 'gai', 'garbage' \
                    , 'garden', 'gel', 'germanstyle', 'gold', 'gourmet' \
                    , 'great', 'northern', 'hair', 'halfmoons', 'halfpint' \
                    , 'hawiian', 'highgluten', 'homestyle', 'hothouse' \
                    , 'house', 'hubbard', 'icecold', 'iceberg', 'idaho' \
                    , 'imitation', 'inchlong', 'inchwide', 'iron', 'jack' \
                    , 'italianblend', 'italianstyle', 'japanese', 'jerk' \
                    , 'johnsonville', 'kellogg', 'kernel', 'kingsford' \
                    , 'lagerstyle', 'land', 'leg', 'lessodium', 'lid' \
                    , 'montreal', 'montrealstyle', 'mission', 'melt' \
                    , 'multi', 'nut', 'ocean', 'spray', 'oven', 'overnight' \
                    , 'overripe', 'palegreen', 'palm', 'paperthin' \
                    , 'parchment', 'pareve', 'pepperidge', 'pilsnerstyle' \
                    , 'pit', 'pkg', 'plain', 'plus', 'pocket', 'porterhouse' \
                    , 'prime', 'rib', 'readymade', 'rise', 'ribeye', 'ribbon' \
                    , 'rotisserie', 'russet', 'vital', 'visible', 'virgin' \
                    , 'extra', 'velvet', 'vegetable', 'unripe', 'uncut' \
                    , 'uncle', 'twocheese', 'twine', 'twist', 'towel' \
                    , 'transfatfree', 'trail', 'transparent', 'tri' \
                    , 'triangle', 'triple', 'tritip', 'tropical', 'turbinado' \
                    , 'udon', 'thai', 'thaistyle', 'texas', 'tie' \
                    , 'threecheese', 'tm', 'tom', 'toothpick', 'torn' \
                    , 'table', 'tablespoonsize', 'sun', 'super', 'sushistyle' \
                    , 'sweet', 'swiss', 'straw', 'storebought', 'stoneground' \
                    , 'visible', 'virgin', 'vital', 'winter', 'wet', 'well' \
                    , 'tvp', 'wet', 'well', 'wellbeaten', 'wellshaken']

    exclude_ingredients = ['cool', 'collard', 'al fresco', 'green', 'pink' \
    , 'black', 'yellow', 'great northern', 'greek', 'creole', 'blue', 'bell' \
    , 'english']
    ingred_caption = []
    exclude_ending_1 = ['y']
    exclude_ending_2 = ['ly', 'ed', 'an']
    exclude_ending_3 = ['ing']
    # iterate over recipes
    for item in ingred_list:
        line_final = []
        # iterate over recipe ingredient line items (in mongo db these did not need to be split)
        for line in item.replace(' or ', ', ').split(','):
            line_str = []
            for word in line.split():
                i=0
                word = word.lower().strip()
                word = word.strip('-')
                word = word.strip('[]')
                word = ''.join(e for e in word if e.isalnum() and not e.isdigit())
                word = wordnet_lemmatizer.lemmatize(word)
                if word not in exclude_words and i < max_ingred_len \
                and word[-1:] not in exclude_ending_1 \
                and word[-2:] not in exclude_ending_2 \
                and word[-3:] not in exclude_ending_3:
                    line_str.append(word)
                    i+=1

            if line_str != [] and line_str[0] not in exclude_ingredients:
                line_final.append(line_str)
        ingred_caption.append(line_final)

    ingred_caption_final = []
    for row in ingred_caption:
        # row_final=['#START#']
        row_final = []
        for item in row:
            item_final = ' '.join(item)
            item_final = item_final.strip('-')

            row_final.append(str(item_final))

        # row_final.append('#END#')

        ingred_caption_final.append(row_final)


    return ingred_caption_final

def create_text_vectorizer(text_list):
    ''' Convert Ingredients to Count Vectors

    Input:
        Raw ingredient list as scraped

    Output:
        Count vector
    '''

    vocab = sorted(set(itertools.chain.from_iterable(text_list)))

    indicoio_keywords = indicoio.keywords(vocab, version=2)
    ingred_caption_keywords = []
    word_keyword = defaultdict(str)

    for i, v in zip(indicoio_keywords, vocab):
        try:
            keyword = str(max(i.iteritems(), key=operator.itemgetter(1))[0])
            word_keyword[v] = keyword
            ingred_caption_keywords.append(keyword)
        except:
            pass

    new_vocab = sorted(set(ingred_caption_keywords))
    corpus = []
    for recipe in text_list:
        for word in recipe:
            corpus.append(word)

    print 'corpus length: ' + str(len(corpus))

    print 'total words: ' + str(len(new_vocab))
    word_indices = dict((c, i) for i, c in enumerate(new_vocab))
    indices_word = dict((i, c) for i, c in enumerate(new_vocab))

    return word_indices, indices_word, word_keyword

def vectorize_text(text_list, word_indices, word_keyword):
    ''' Vectorize multiple cleaned lists of ingredients '''
    y = np.zeros((len(text_list), len(word_indices)), dtype=np.bool)
    for i, recipe in enumerate(text_list):
        for t, word in enumerate(recipe):
            keyword = word_keyword[word]
            if keyword != '':
                y[i, word_indices[keyword]] = 1
    return y

def tensorize_text(text_list, word_indices, max_caption_len):
    y = np.zeros((len(text_list), len(word_indices)), dtype=np.bool)
    X = np.zeros((len(sentences), max_caption_len, len(chars)), dtype=np.bool)

    for i, recipe in enumerate(text_list):
        for t, word in enumerate(recipe):
            X[i, t, word_indices[word]] = 1
        y[i, t, word_indices[text_list[i+1]]] = 1
    return X, y

if __name__ == '__main__':
    base_path = '../'
    df = pd.read_csv(base_path+'data/recipe_data.csv')
    # vectorizer, words = vectorize_text(df['ingred_list'], 1000)
    text_list = clean_text(df['ingred_list'])

    word_indices, indices_word, word_keyword = create_text_vectorizer(text_list)
