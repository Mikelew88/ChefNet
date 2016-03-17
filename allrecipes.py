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


def Pull_Recipe_Links(start_url='http://allrecipes.com/recipes/', limit=99999):

    Page_Error = False
    current_url = start_url
    i = 1
    Recipe_Links = defaultdict(str)

    while not Page_Error and i<limit:
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

    return Recipe_Links

def scrape_ingredients(link):
    url = "http://allrecipes.com"+link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data)

    image_page_link = soup.findAll('a', {'class':'icon-photoPage'})[0].get('href')

    ingreds = []
    for s in soup.findAll('li', {'class': 'checkList__line'}):
        ingred = s.text.strip()
        if not ingred.startswith('Add') and not ingred.startswith('ADVERTISEMEN'):
            ingreds.append(ingred[:s.text.strip().find('\n')])

    return ingreds, image_page_link

 def scrape_photos(link, recipe_num, num_photos = 25):
     url = "http://allrecipes.com"+link
     data = urllib2.urlopen(url).read()
     soup = BeautifulSoup(data)

     i=0

     for img in soup.findAll('img'):
         src = str(img.get('src'))
         if src[-4:]=='.jpg' and i < num_photos:
             # pdb.set_trace()
             print i
             i+=1
            #  urllib.urlretrieve(src, 'images/Recipe_Images/'+item+str(i)+'.jpg')

     pass

def store_data(image_url, filename):
    mime_type = mimetypes.guess_type(image_url)[0]

    a = fs.put(image_url, contentType=mime_type, filename)

def run_scrapper():
    """define opener"""
    class MyOpener(urllib.FancyURLopener):
        version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
    myopener = MyOpener()

    #run mongod first!

    db_client = MongoClient()
    db = db_client['allrecipes']

    fs = gridfs.GridFS(db)

    Recipe_Links = Pull_Recipe_Links(limit = 10)

    for key, val in Recipe_Links.iteritems():
        scrape_ingredients(val[0])


if __name__ == '__main__':
