# Welcome to ChefNet

Here you may find my Galvanize Capstone Project.

__Outline__

* [Data](#data-wrangling)
* [Ingredient Categorization](#ingredient-label-wrangling)
* [Image Processing](#image-processing)
* [Neural Network Architecture](#neral-network-architecture)

### Motivation

What can we teach computer's about food? ChefNet is a convolutional neural network that identifies the ingredients in a dish by analyzing a picture of it.

### Data

To create a labeled dataset of images of food, I scraped the recipes and user submitted photos of 17,000 recipes, totaling 230,000 user photos, from allrecipes.com. The scraper was run on an AWS instance and took about a day to run, massive speedups may be attributed to parallelizing and threading the scraping process. Code is in the __Web_scrapers__ folder under Scripts.

### Ingredient label wrangling

In order to train a neural net, I needed to create consistent labels for ingredients. I took two approaches. My first approach was to start with the scraped list of ingredients, and identify the keyword using the indico keyword extraction api, while iteratively remove all words not critical to the underlying food item. My second approach, which I ultimately used to train my net, was to start with a cleaned list of ingredients initially scraped from   enchantedlearning.com

It is critical that image labels are as cleans as possible, otherwise the neural network will have difficulty learning. I think it is important to allow the model to have multi-word labels to represent items such as bell pepper. A useful extension of this project may be to vectorize the labels, so that the net will learn that the ingredient similarity, for example, misclassifying beef and steak is closer than chicken and peas. Vectorization methods require tokenization of text first. This may be an area worth explore further.

### Image Processing

Neural networks were train with raw image data, and convolved imaged data that was passed though the 2014 image net winner, VGG-16 from Oxford. Transfer learning proved more fruitful given the limited size of my dataset. Activations were taken at the end of layer 30, before flattening to dense layers. It would be interesting to compare results using activations taken after these dense layers, but I did not have time to explore this comparison. 

Images were downsized to 100x100 so that I could iterate through training multiple models, in the time allotted for capstone projects.

### Neural Network Architecture



### Thank you

Big thank you to Jesse Lieman-Sifry for inspiring this idea, as well as to my Galvanize Instructors and peers for continuous help and suggestions along the way. It was a pleasure to work with all of you.
