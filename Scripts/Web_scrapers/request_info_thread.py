''' Code built upon code Sean Sall. Thanks Sean! '''

from threading import Thread
import urllib, urllib2
from bs4 import BeautifulSoup


class request_info_thread(Thread):
    '''
    Inherits from Thread so I can store results of my threading.
    I want to be able to return something from my threading. This
    class will allow me to do that - perform all of the requesting
    and data gathering that I want to do, but store the results on
    the class so that I can access them later.
    '''

    def __init__(self, recipe_id, link):
        super(RequestInfoThread, self).__init__()
        self.recipe_id = recipe_id
        self.url = 'http://allrecipes.com'+link
        data = urllib2.urlopen(self.url).read()
        self.soup = BeautifulSoup(data, 'lxml')
        self.json_dct = None

    def run(self):
        self.json_dct = self._request_info()
        if self.json_dct:
            self.image_url = self._get_img_url()
            img_data = urllib2.urlopen(self.image_url).read()
            self.img_soup =  BeautifulSoup(img_data, 'lxml')
            self._scrape_photos(num_photos=999)

    def _request_info(self):
        '''
        Grab relevant information from the row and store it in mongo.
        Make sure that if there is missing information that is not crucial to my analysis, we still store the data.
        By checking the Mongo table during the extraction process we can save time by not getting the html of the url if that url already exists in the table.
        '''

        item_name = self.soup.find('h1', {'class':'recipe-summary__h1'}).text

        submitter_name = self.soup.find('span', {'class':'submitter__name'}).text
        submitter_desc = self.soup.find('div', {'class':'submitter__description'}).text
        stars = self.soup.find('div', {'class':'rating-stars'}).get('data-ratingstars')

        ingred_list = []

        for s in self.soup.findAll('li', {'class': 'checkList__line'}):
            ingred = s.text.strip()
            if not ingred.startswith('Add') and not ingred.startswith('ADVERTISEMEN'):
                ingred_list.append(ingred[:s.text.strip().find('\n')])

        dircetions = self.soup.find('div', {'class':'directions--section'})

        #If missing time data, set var to null
        prep_time = null_time_helper(dircetions.find('time', {'itemprop':'prepTime'}))
        cook_time = null_time_helper(dircetions.find('time', {'itemprop':'cookTime'}))
        total_time = null_time_helper(dircetions.find('time', {'itemprop':'totalTime'}))

        directions = dircetions.findAll('span', {'class':'recipe-directions__list--item'})
        direction_list = [d.text for d in directions]

        #Throw data into MongoDB
        json_dct = ({'id':self.recipe_id, 'item_name': item_name, 'ingred_list':ingred_list, 'direction_list':direction_list, 'stars': stars, 'submitter_name':submitter_name, 'submitter_desc': submitter_desc, 'prep_time':prep_time, 'cook_time':cook_time, 'total_time':total_time})

        return json_dct

    def _get_img_url(self):
        image_url = self.soup.findAll('a', {'class':'icon-photoPage'})[0].get('href')
        return 'http://allrecipes.com'+image_url

    def _scrape_photos(self, num_photos = 25):
        i=0

        img_band = self.img_soup.find('ul', {'class':'photos--band'})
        for img in img_band.findAll('img'):
            src = str(img.get('src'))
            if src[-4:]=='.jpg' and i < num_photos:
                urllib.urlretrieve(src, '/data/Recipe_Images_2/'+str(self.recipe_id)+'_'+str(i)+'.jpg')
                i+=1
        pass

def null_time_helper(func):
    ''' This will allow me to store pages that do not contain all scraped fields

    Input:  function

    Output: value of function or None
    '''
    try:
        val = func.get('datetime')
    except:
        val = None
    return val

if __name__ == '__main__':
    test = request_info_thread(211166, '/recipe/211166/ham-and-broccoli-bake/?internalSource=rotd&referringId=1&referringContentType=recipe%20hub')
    test.start()
