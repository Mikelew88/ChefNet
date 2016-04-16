import os
from skimage.io import imread

def remove_bad_jpgs(dir_path='/data/Recipe_Images/', size = .001):
    ''' Go through images and remove empty files before preprocessing '''

    img_dir = os.listdir(dir_path)

    for jpg in img_dir:
        jpg_path = dir_path+jpg
        if os.path.getsize(jpg_path) < size:
            print 'Corrupted File: {}'.format(jpg_path)
            os.rename(jpg_path, "/data/Bad_Images/"+jpg)

def remove_empty_jpgs(dir_path='/data/Recipe_Images/'):
    ''' Go through images and remove blank and/or black and white images '''
    img_dir = os.listdir(dir_path)

    for jpg in img_dir:
        jpg_path = dir_path+jpg
        try:
            img = imread(jpg_path)
            if img.shape != (250, 250, 3):
                print 'Empty File: {}'.format(jpg_path)
                os.rename(jpg_path, "/data/Bad_Images/"+jpg)
        except IOError:
            print 'Corrupted File: {}'.format(jpg_path)
            os.rename(jpg_path, "/data/Really_Bad_Images/"+jpg)

if __name__ == '__main__':
    # remove_bad_jpgs(size = 3)
    remove_empty_jpgs()
