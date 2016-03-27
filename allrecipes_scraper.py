import pandas as pd
import datetime

from bs4 import BeautifulSoup
import urllib, urllib2
from selenium import webdriver
import re

from pymongo import MongoClient
import gridfs
import mimetypes
import multiprocessing
import threading

import pdb

def Pull_Recipe_Links(i):
    """define opener"""
    class MyOpener(urllib.FancyURLopener):
        version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
    myopener = MyOpener()

    url = "http://allrecipes.com/recipes/?page=" + str(i)

    page = myopener.open(url)
    soup = BeautifulSoup(page, 'lxml')

    Link_Soup = soup.findAll('a')

    for a in Link_Soup:
        link = str(a.get('href')).strip()
        scrape_item(link)

def scrape_item(link):
    #Parse url string to locate recipe name and number
    end_recipe_number = link[8:].find('/')+8
    recipe_id = link[8:end_recipe_number]
    recipe_label = link[end_recipe_number+1:link[end_recipe_number+1:].find('/')+end_recipe_number+1]

    #Store recipe information in a default dictionary with the recipe number as the key (in case there are duplicate recipe names)

    if link[:8]=='/recipe/':

        # scrape_recipe_page(recipe_id, link)

        # For threading
        jobs=[]
        t = threading.Thread(target=scrape_recipe_page, args=(recipe_id,link))
        jobs.append(t)
        t.start()

    pass

def scrape_recipe_page(recipe_id, link):

    #Spin up mongo data, run mongod first!
    # fs = gridfs.GridFS(db)
    # db.remove({})

    db_client = MongoClient()
    db = db_client['allrecipes']

    url = "http://allrecipes.com"+link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data, 'lxml')
    recipe_db = db.recipe_data

    # Scrape a bunch of data

    #Initialize all vars to None

    stars = soup.find('div', {'class':'rating-stars'}).get('data-ratingstars')
    submitter_name = soup.find('span', {'class':'submitter__name'}).text
    submitter_desc = soup.find('div', {'class':'submitter__description'}).text
    item_name = soup.find('h1', {'class':'recipe-summary__h1'}).text

    ingred_list = []

    for s in soup.findAll('li', {'class': 'checkList__line'}):
        ingred = s.text.strip()
        if not ingred.startswith('Add') and not ingred.startswith('ADVERTISEMEN'):
            ingred_list.append(ingred[:s.text.strip().find('\n')])

    dircetions = soup.find('div', {'class':'directions--section'})

    prep_time = dircetions.find('time', {'itemprop':'prepTime'}).get('datetime')
    cook_time = dircetions.find('time', {'itemprop':'cookTime'}).get('datetime')
    total_time = dircetions.find('time', {'itemprop':'totalTime'}).get('datetime')

    directions = dircetions.findAll('span', {'class':'recipe-directions__list--item'})
        direction_list = [d.text for d in directions]

        #Need selenium to scrap related categories

        # driver = webdriver.Firefox()
        # driver.get(url)
        # sel_soup = BeautifulSoup(driver.page_source, 'html.parser')
        #
        # cat_soup = sel_soup.find('div', {'class': ['tab-pane', 'ng-scope', 'ng-isolate-scope'],'title':'Categories'})
        # cats = cat_soup.findAll('h3', {'class':'grid-col__h3'})
        # cat_text = [cat.text for cat in cats]

    #Throw data into MongoDB
    recipe_db.insert_one({'id':recipe_id, 'item_name': item_name, 'ingred_list':ingred_list, 'direction_list':direction_list, 'stars': stars, 'submitter_name':submitter_name, 'submitter_desc': submitter_desc, 'prep_time':prep_time, 'cook_time':cook_time, 'total_time':total_time})
    # , 'cat_text':cat_text})

    #Scrape Images
    image_page_link = soup.findAll('a', {'class':'icon-photoPage'})[0].get('href')
    scrape_photos(recipe_id, image_page_link)
    pass

def scrape_photos(recipe_id, image_page_link, num_photos = 25):
    url = "http://allrecipes.com"+image_page_link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data, 'lxml')

    i=0

    image_links = []

    for img in soup.findAll('img'):
        src = str(img.get('src'))
        if src[-4:]=='.jpg' and i < num_photos:
            image_links.append(src)

            # mime_type = mimetypes.guess_type(image_page_link)[0]

            urllib.urlretrieve(src, 'images/Recipe_Images/'+str(recipe_id)+'_'+str(i)+'.jpg')
    pass

def run_parallel(num_pages = 10):
    page_range = range(1,num_pages)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    outputs = pool.map(Pull_Recipe_Links, page_range)

    pass

if __name__ == '__main__':
    # recipe_dict = Pull_Recipe_Links(limit = 9999)

    # store_data(recipe_dict)
    # run_parallel(3)

    #Testing
    Pull_Recipe_Links(3)
    # img_url = '/recipe/15925/creamy-au-gratin-potatoes/photos/738814/'
    # scrape_photos(15925, img_url, 4)
