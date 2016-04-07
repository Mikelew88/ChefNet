import os
from skimage.io import imread

def remove_bad_jpgs(dir_path='/data/Recipe_Images/'):
    '''
    Go through images and remove empty files before preprocessing
    '''

    img_dir = os.listdir(dir_path)

    for jpg in img_dir:
        jpg_path = dir_path+jpg
        if os.path.getsize(jpg_path) < .001:
            print 'Corrupted File: {}'.format(jpg_path)
            os.rename(jpg_path, "/data/Bad_Images/"+jpg)

def remove_empty_jpgs(dir_path='/data/Recipe_Images/'):
    '''
    Go through images and remove blank images
    '''
    img_dir = os.listdir(dir_path)

    for jpg in img_dir:
        jpg_path = dir_path+jpg
        img = imread(jpg_path)
        if img.shape != (250, 250, 3):
            print 'Empty File: {}'.format(jpg_path)
            os.rename(jpg_path, "/data/Bad_Images/"+jpg)


if __name__ == '__main__':
    remove_empty_jpgs()
