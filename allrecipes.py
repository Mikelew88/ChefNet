import pandas as pd
import datetime

from bs4 import BeautifulSoup
import urllib, urllib2
import re

from pymongo import MongoClient
import gridfs
import mimetypes
import multiprocessing
import threading

import pdb

def Pull_Recipe_Links(page):
    """define opener"""
    class MyOpener(urllib.FancyURLopener):
        version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
    myopener = MyOpener()

    url = "http://allrecipes.com/recipes/?page=" + str(i)

    page = myopener.open(url)
    soup = BeautifulSoup(page)

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

        scrape_sub_page(recipe_id, link)

    return recipe_dict


def scrape_sub_page(recipe_id, link):

    url = "http://allrecipes.com"+link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data)
    ingred_db = db.ingreds

    image_page_link = soup.findAll('a', {'class':'icon-photoPage'})[0].get('href')
    scrape_photos(image_page_link, )

    ingred_list = []

    stars = soup.find('div', {'class':'rating-stars'}).get('data-ratingstars')

    submitter_name = soup.find('span', {'class':'submitter__name'}).text

    submitter_desc = soup.find('div', {'class':'submitter__description'}).text

    item_name = soup.find('h1', {'class':'recipe-summary__h1'}).text

    for s in soup.findAll('li', {'class': 'checkList__line'}):
        ingred = s.text.strip()
        if not ingred.startswith('Add') and not ingred.startswith('ADVERTISEMEN'):
            ingred_list.append(ingred[:s.text.strip().find('\n')])

    dircetions = soup.find('div', {'class':'directions--section'})

    prep_time = dircetions.find('time', {'itemprop':'prepTime'}).get('datetime')
    cook_time = dircetions.find('time', {'itemprop':'cookTime'}).get('datetime')
    total_time = = dircetions.find('time', {'itemprop':'totalTime'}).get('datetime')

    dircetions = dircetions.findAll('span', {'class':'recipe-directions__list--item'})
    direction_list = [d.text for d in directions]

    cat_soup = soup.find('div', {'class': ['tab-pane', 'ng-scope', 'ng-isolate-scope'],'title':'Categories'})
    cats = cat_soup.findAll('h3', {'class':'grid-col__h3'})
    cat_text = [cat.text for cat in cats]

    ingred_db.insert({'id':recipe_id, 'item_name': item_name, 'ingred_list':ingred_list, 'direction_list':direction_list, 'stars': stars, 'submitter_name':submitter_name, 'submitter_desc': submitter_desc, 'prep_time':prep_time, 'cook_time':cook_time, 'total_time':total_time, 'cat_text':cat_text})
    pass

def scrape_photos(image_page_link, recipe_id, num_photos = 25):
    url = "http://allrecipes.com"+image_page_link
    data = urllib2.urlopen(url).read()
    soup = BeautifulSoup(data)

    i=0

    image_links = []

    for img in soup.findAll('img'):
        src = str(img.get('src'))
        if src[-4:]=='.jpg' and i < num_photos:
            image_links.append(src)

            mime_type = mimetypes.guess_type(image_url)[0]

            urllib.urlretrieve(src, 'images/Recipe_Images/'+recipe_id+'_'+str(i)+mime_type)
            print i
            i+=1
    pass

def run_parallel(num_pages = 10):
    #run mongod first!

    db_client = MongoClient()
    db = db_client['allrecipes']

    # fs = gridfs.GridFS(db)
    db.remove({})

    page_range = range(1,num_pages)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    outputs = pool.map(Pull_Recipe_Links, page_range)

    # For threading
    # jobs=[]
    # for business in ids['businesses']:
    #     business_path = BUSINESS_PATH + business['id']
    #     t = threading.Thread(target=request_business, args=(business_path,))
    #     jobs.append(t)
    #     t.start()
    pass

if __name__ == '__main__':
    recipe_dict = Pull_Recipe_Links(limit = 9999)

    store_data(recipe_dict)
