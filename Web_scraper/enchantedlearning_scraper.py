from bs4 import BeautifulSoup

import urllib

def scrape_text(url):
    ''' Quick scraper to grab potential list of ingredients '''

    """define opener"""
    class MyOpener(urllib.FancyURLopener):
        version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'
    myopener = MyOpener()

    page = myopener.open(url)
    soup = BeautifulSoup(page, 'lxml')

    tables = soup.find_all('tr', {'align': 'center', 'valign':'top'})

    food_labels = []
    for i in tables:
        text = i.get_text()
        text = text.split('\n')
        for j in text:
            text = str(text).strip().strip('\').lower()
            if text == 'adjectives':


                return food_labelss
            food_labels.append(text)

    return food_labels

if __name__ == '__main__':

    food_labels = scrape_text('http://www.enchantedlearning.com/wordlist/food.shtml')
