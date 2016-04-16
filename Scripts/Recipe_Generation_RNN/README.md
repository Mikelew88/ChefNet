Here you may find a Recurrent Neural Network that generates recipes character by character. The ingredient lists scraped from allrecipes.com have been heavily filtered to allow the network to produce a list of food items that may be interesting. Perhaps it would be fun to try to make a dish based on what it comes up with!

To create labels from the raw data I took the following steps:

1. I removed all numbers and punctuation

1. I lemmatized the text

1. I utilized indico's keyword extraction api to assign values to the words or sets of words in the text, then I took the argmax

1. I created a massive list of exclusion words which may be seen in __preprocess_text__ under the Recipe_Generation_RNN folder in Scrips

The __lstm_text_generation__ code is highly borrowed from Kera's docs.
