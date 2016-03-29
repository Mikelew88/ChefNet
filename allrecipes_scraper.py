import pandas as pd
import datetime

from bs4 import BeautifulSoup
import urllib, urllib2
# from selenium import webdriver
# import re

from pymongo import MongoClient
# import gridfs
# import mimetypes
import multiprocessing
from threading import Thread
from RequestInfoThread import RequestInfoThread

def Pull_Recipe_Links(i):
    """define opener"""
    class MyOpener(urllib.FancyURLopener):
        version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
    myopener = MyOpener()

    #Store results in mongo
    db_client = MongoClient()
    db = db_client['allrecipes']
    recipe_db = db['recipe_data']
    # recipe_db.remove({})

    url = "http://allrecipes.com/recipes/?page=" + str(i)

    page = myopener.open(url)
    soup = BeautifulSoup(page, 'lxml')

    Link_Soup = set(soup.findAll('a'))

    for a in Link_Soup:
        link = str(a.get('href')).strip()

        if link[:8]=='/recipe/':
            mongo_update_lst = scrape_search(link, recipe_db)
            if mongo_update_lst:
                store_data(mongo_update_lst, recipe_db)

    db_client.close()


def scrape_search(link, recipe_db):
    #Parse url string to locate recipe name and number
    end_recipe_number = link[8:].find('/')+8
    recipe_id = link[8:end_recipe_number]
    recipe_label = link[end_recipe_number+1:link[end_recipe_number+1:].find('/')+end_recipe_number+1]

    if already_exists(recipe_db, recipe_id):
        return False

    # Thread scrape_recipe_page(recipe_id, link)
    threads=[]
    mongo_update_lst = []

    t = RequestInfoThread(recipe_id,link)

    t.start()
    threads.append(t)

    for thread in threads:
        thread.join()
        mongo_update_lst.append(thread.json_dct)

    return mongo_update_lst

def store_data(mongo_update_lst, recipe_db):
    for json_dct in mongo_update_lst:
        if json_dct:
            recipe_db.insert_one(json_dct)
    pass

def already_exists(recipe_db, id):
    return bool(recipe_db.find({'id': id}).count())

def run_parallel(num_pages = 10):

    page_range = range(1,num_pages)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    outputs = pool.map(Pull_Recipe_Links, page_range)

    pass

if __name__ == '__main__':
    # store_data(recipe_dict)
    run_parallel(num_pages=999999)

    #Spin up mongo data, run mongod first!

    # for i in xrange(1,3):
    
    # Pull_Recipe_Links(i)
