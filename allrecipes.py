import pandas as pd
import datetime
from collections import defaultdict

from bs4 import BeautifulSoup
import urllib, urllib2
import re

from pymongo import MongoClient
import gridfs
import mimetypes

import pdb

def Pull_Recipe_Links(start_url='http://allrecipes.com/recipes/', limit=99999):
    """define opener"""
    class MyOpener(urllib.FancyURLopener):
        version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
    myopener = MyOpener()

    Page_Error = False
    current_url = start_url
    i = 1
    recipe_dict = defaultdict(str)

    while not Page_Error and i<limit:

        # try:

        page = myopener.open(current_url)
        soup = BeautifulSoup(page)

        Link_Soup = soup.findAll('a')

        for a in Link_Soup:
            link = str(a.get('href')).strip()

            #Parse url string to locate recipe name and number
            end_recipe_number = link[8:].find('/')+8
            recipe_number = link[8:end_recipe_number]
            recipe_name = link[end_recipe_number+1:link[end_recipe_number+1:].find('/')+end_recipe_number+1]

            #Store recipe information in a default dictionary with the recipe number as the key (in case there are duplicate recipe names)

            # pdb.set_trace()
#
            print link
            if link[:8]=='/recipe/':

                ingreds, image_page_link = scrape_ingredients(link)

                image_links = scrape_photos(image_page_link)

                recipe_dict[recipe_number] = [recipe_name, link, ingreds, image_links]

        i += 1
        current_url = "http://allrecipes.com/recipes/?page=" + str(i)

        # except Exception:
        #     Page_Error = True

    return recipe_dict

def scrape_ingredients(link):

    url = "http://allrecipes.com"+link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data)
    # pdb.set_trace()

    image_page_link = soup.findAll('a', {'class':'icon-photoPage'})[0].get('href')

    ingreds = []
    for s in soup.findAll('li', {'class': 'checkList__line'}):
        ingred = s.text.strip()
        if not ingred.startswith('Add') and not ingred.startswith('ADVERTISEMEN'):
            ingreds.append(ingred[:s.text.strip().find('\n')])

    return ingreds, image_page_link

def scrape_photos(image_page_link, num_photos = 25):
    url = "http://allrecipes.com"+image_page_link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data)

    i=0

    image_links = []

    for img in soup.findAll('img'):
        src = str(img.get('src'))
        if src[-4:]=='.jpg' and i < num_photos:
            image_links.append(src)
            print i
            i+=1

    return image_links

def store_data(recipe_dict):
    mime_type = mimetypes.guess_type(image_url)[0]

    a = fs.put(image_url, contentType=mime_type, filename=str('temp_imgname'+'.jpg'))

def run_scrapper():
    #run mongod first!

    db_client = MongoClient()
    db = db_client['allrecipes']

    fs = gridfs.GridFS(db)

    recipe_dict = Pull_Recipe_Links(limit = 10)
    return recipe_dict

if __name__ == '__main__':
    recipe_dict = run_scrapper()
