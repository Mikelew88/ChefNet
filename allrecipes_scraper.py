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

    #Spin up mongo data, run mongod first!
    # fs = gridfs.GridFS(db)
    # db.remove({})
    global db
    db_client = MongoClient()
    db = db_client['allrecipes']


    url = "http://allrecipes.com/recipes/?page=" + str(i)

    page = myopener.open(url)
    soup = BeautifulSoup(page, 'lxml')

    Link_Soup = soup.findAll('a')

    for a in Link_Soup:
        link = str(a.get('href')).strip()
        scrape_search(link)


def scrape_search(link):
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

def null_time_helper(func):
    '''
    Input: function
    Output: value of function or None
    This will allow me to store pages that do not contain all scraped fields.
    '''
    try:
        val = func.get('datetime')
    except:
        val = None
    return val

def scrape_recipe_page(recipe_id, link):
    '''
    INPUT: Unique Recipe ID, Link to individual Recipe Page
    OUTPUT:
    bool on whether extration was successful or not (Will also return False if the url already exists in the Mongo table, if the source is no longer availible on allrecipes, or the section is something we don't care about)
    Dict to insert into Mongo Database or empty string if it isn't something we want to insert into Mongo
    By checking the Mongo table during the extraction process we can save time by not getting the html of the url if that url already exists in the table.
    '''

    url = "http://allrecipes.com"+link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data, 'lxml')

    #Create cursor for each thread
    recipe_db = db.recipe_data

    # Scrape a bunch of data

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

    #If missing time data, set var to null
    prep_time = null_time_helper(dircetions.find('time', {'itemprop':'prepTime'}))
    cook_time = null_time_helper(dircetions.find('time', {'itemprop':'cookTime'}))
    total_time = null_time_helper(dircetions.find('time', {'itemprop':'totalTime'}))

    directions = dircetions.findAll('span', {'class':'recipe-directions__list--item'})
    direction_list = [d.text for d in directions]

    #Throw data into MongoDB
    recipe_db.insert_one({'id':recipe_id, 'item_name': item_name, 'ingred_list':ingred_list, 'direction_list':direction_list, 'stars': stars, 'submitter_name':submitter_name, 'submitter_desc': submitter_desc, 'prep_time':prep_time, 'cook_time':cook_time, 'total_time':total_time})

    #Scrape Images
    image_page_link = soup.findAll('a', {'class':'icon-photoPage'})[0].get('href')
    scrape_photos(recipe_id, image_page_link)
    pass

def scrape_photos(recipe_id, image_page_link, num_photos = 25):
    url = "http://allrecipes.com"+image_page_link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data, 'lxml')

    i=0

    img_band = soup.find('ul', {'class':'photos--band'})
    for img in img_band.findAll('img'):
        src = str(img.get('src'))
        if src[-4:]=='.jpg' and i < num_photos:
            urllib.urlretrieve(src, 'images/Recipe_Images/'+str(recipe_id)+'_'+str(i)+'.jpg')
            i+=1
    pass

def run_parallel(num_pages = 10):

    page_range = range(1,num_pages)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    outputs = pool.map(Pull_Recipe_Links, page_range)

    pass

if __name__ == '__main__':
    # recipe_dict = Pull_Recipe_Links(limit = 9999)

    # store_data(recipe_dict)
    # run_parallel(num_pages=3)

    #Testing
    # Pull_Recipe_Links(3)
    # img_url = '/recipe/15925/creamy-au-gratin-potatoes/photos/738814/'
    # scrape_photos(15925, img_url, 4)

    for i in xrange(1,9):
        Pull_Recipe_Links(i)
