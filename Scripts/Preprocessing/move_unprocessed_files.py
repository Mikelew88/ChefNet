import os
from shutil import copy

if __name__ == '__main__':

    processed_dir = [x.split('.')[0] for x in os.listdir('/data/preprocessed_imgs/')]
    unprocessed_dir = [x.split('.')[0] for x in os.listdir('/data/Recipe_Images/')]

    outer_union = list(set(unprocessed_dir)-set(processed_dir))
