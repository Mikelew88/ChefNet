import pandas as pd
import datetime
from collections import defaultdict
import pdb

from bs4 import BeautifulSoup
import urllib, urllib2
import re

from pymongo import MongoClient
import gridfs
import mimetypes


def Pull_Recipe_Links(start_url='http://allrecipes.com/recipes/'):

    Page_Error = False
    current_url = start_url
    i = 1
    Recipe_Links = defaultdict(str)

    while not Page_Error and i<10:
        print(i)
        try:

            page = myopener.open(current_url)
            soup = BeautifulSoup(page)

            Link_Soup = soup.findAll('a')

            for a in Link_Soup:
                link = str(a.get('href')).strip()

                end_recipe_number = link[8:].find('/')+8
                recipe_number = link[8:end_recipe_number]
                recipe_name = link[end_recipe_number+1:link[end_recipe_number+1:].find('/')+end_recipe_number+1]

                if link[:8]=='/recipe/':
                    Recipe_Links[recipe_number] = [link, recipe_name]

            i += 1
            current_url = "http://allrecipes.com/recipes/?page=" + str(i)

        except Exception:
            Page_Error = True

    return Recipe_Links, Page_Error

def scrape_ingredients(link):
    url = "http://allrecipes.com"+link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data)

    image_link = soup.findAll('a', {'class':'icon-photoPage'})[0].get('href')

    #save photos


    ingreds = []
    for s in soup.findAll('li', {'class': 'checkList__line'}):
        ingred = s.text.strip()
        if not ingred.startswith('Add') and not ingred.startswith('ADVERTISEMEN'):
            ingreds.append(ingred[:s.text.strip().find('\n')])

    return ingreds, image_link

def store_data():
    #run mongod first!

    db_client = MongoClient()
    db = db_client['allrecipes']

    fs = gridfs.GridFS(db)

    image_url = 'http://images.media-allrecipes.com/userphotos/600x600/3371109.jpg'

    mime_type = mimetypes.guess_type(image_url)[0]

    a = fs.put(image_url, contentType=mime_type, filename='test.jpg')

if __name__ == '__main__':
    """define opener"""
    class MyOpener(urllib.FancyURLopener):
        version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
    myopener = MyOpener()

    #Scrapes all recipe links

    # time = datetime.datetime.utcnow()

    # url = '/recipe/10402/the-best-rolled-sugar-cookies/photos/3326435/'

    # save_photos(url,'the-best-rolled-sugar-cookies')



    db_client = MongoClient()
    db = db_client['allrecipes']

    fs = gridfs.GridFS(db)

    image_url = 'http://images.media-allrecipes.com/userphotos/600x600/3371109.jpg'

    mime_type = mimetypes.guess_type(image_url)[0]

    a = fs.put(image_url, contentType=mime_type, filename='test.jpg')
    # Recipe_Links, Page_Error = Pull_Recipe_Links()
    # ingreds, image_link=scrape_ingredients('/recipe/10402/the-best-rolled-sugar-cookies/')

    # http://allrecipes.com/recipe/6698/moms-zucchini-bread/
    # imgUrl = 'http://images.media-allrecipes.com/userphotos/250x250/3326435.jpg'
    # urllib.urlretrieve(imgUrl, os.path.basename(imgUrl))

    # MONGODB_HOST = 'localhost'
    # MONGODB_PORT = 27017
    #
    # mongo_con = MongoClient(MONGODB_HOST, MONGODB_PORT)
    # grid_fs = gridfs.GridFS(mongo_con.allrecipe_database)
