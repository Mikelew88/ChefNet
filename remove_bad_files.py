import os

def remove_bad_jpgs(dir_path='/data/Recipe_Images/'):
    '''
    Go through images and remove empty files before preprocessing

    Input:
        Image Directory location
    Output:
        N/a
    '''

    img_dir = os.listdir(dir_path)

    for jpg in img_dir:
        jpg_path = dir_path+jpg
        if os.path.getsize(jpg_path) < .001:
            os.path.getsize(jpg_path)
            print 'Corrupted File: {}'.format(jpg_path)
            os.rename(jpg_path, "/data/Bad_Images/"+jpg)

if __name__ == '__main__':
    remove_bad_jpgs()
