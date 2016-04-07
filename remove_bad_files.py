import os

def remove_bad_jpgs(dir_path='/data/Recipe_Images'):
    '''
    Go through images and remove empty files before preprocessing

    Input:
        Image Directory location
    Output:
        N/a
    '''

    img_dir = os.listdir(dir_path)

    for jpg in img_dir:
        if os.path.getsize(dir_path+jpg) < .001:
            os.path.getsize('C:\\Python27\\Lib\\genericpath.py')
            print 'Corrupted File: {}'.format(file_loc)
            os.rename(file_loc, "data/Bad_Images/"+file_loc.split('/')[-1])
            bad_images.append(file_loc)

if __name__ == '__main__':
    remove_bad_jpgs()
