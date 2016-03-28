import pandas as pd
import datetime

from bs4 import BeautifulSoup
import urllib, urllib2
# from selenium import webdriver
# import re

from pymongo import MongoClient
# import gridfs
# import mimetypes
# import multiprocessing
from threading import Thread
from RequestInfoThread import RequestInfoThread

# import pdb

def Pull_Recipe_Links(i):
    """define opener"""
    class MyOpener(urllib.FancyURLopener):
        version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
    myopener = MyOpener()


    # fs = gridfs.GridFS(db)
    # db.remove({})

    url = "http://allrecipes.com/recipes/?page=" + str(i)

    page = myopener.open(url)
    soup = BeautifulSoup(page, 'lxml')

    Link_Soup = soup.findAll('a')

    for a in Link_Soup:
        link = str(a.get('href')).strip()
        mongo_update_lst = scrape_search(link)
        store_data(mongo_update_lst)


def scrape_search(link):
    #Parse url string to locate recipe name and number
    end_recipe_number = link[8:].find('/')+8
    recipe_id = link[8:end_recipe_number]
    recipe_label = link[end_recipe_number+1:link[end_recipe_number+1:].find('/')+end_recipe_number+1]

    #Store recipe information in a default dictionary with the recipe number as the key (in case there are duplicate recipe names)

    # scrape_recipe_page(recipe_id, link)
    # For threading
    threads=[]
    mongo_update_lst = []

    if link[:8]=='/recipe/':


        t = RequestInfoThread(recipe_id,link)
        t.start()
        threads.append(t)

        for thread in threads:
            thread.join()
            mongo_update_lst.append(thread.json_dct)

    return mongo_update_lst

def store_data(mongo_update_lst):
    #Store results in mongo
    db_client = MongoClient()
    db = db_client['test']
    recipe_db = db['recipe_data']

    for json_dct in mongo_update_lst:
        for k, v in json_dct.iteritems():
            if v:
                recipe_db.update_one({'id': k}, {'$set':{k:v}})
    db_client.close()
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

    #Spin up mongo data, run mongod first!

    for i in xrange(1,3):
        Pull_Recipe_Links(i)
