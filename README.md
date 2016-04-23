# Welcome to ChefNet

What can we teach computer's about food? ChefNet is a convolutional neural network that identifies the ingredients in a dish by analyzing a picture of it.

For example, what ingredients do you think are in this delicious looking cake?

<img src="images/carrot_cake.jpg" width="300">
<br><br>

# Table of Contents

1. [Data](#data)
    * [Ingredient Labeling](#ingredient-label-wrangling)
    * [Image Processing](#image-processing)
    * [Neural Network Architecture](#neural-network-architecture)
1. [Results](#results)
    * [Overall Results](#overall-results)
    * [Example Predictions](#example-predictions)
    * [Data Issues](#data-issues)
1. [Try ChefNet yourself](#try-chefnet-yourself)
1. [Next Steps](#next-steps)
1. [Recipe Generation](#recipe-generation-with-recurrent-neural-networks)
1. [Thank you](#thank-you)
    * [References](#references)

# Data

To create a labeled dataset of images of food, I scraped the recipes and user submitted photos of 17,000 recipes, totaling 230,000 user photos, from allrecipes.com. The scraper was run on an AWS instance and took about a day to run. Massive speedups may be attributed to parallelizing and threading the scraping process. Code is in the [Web Scrapers](/Scripts/Web_scrapers) folder under Scripts.

### Ingredient label wrangling

<img src="figures/vocab_wordcloud.png" width="400">

In order to train a neural net, I needed to create consistent labels for ingredients. I took two approaches. My first approach was to start with the scraped list of ingredients, and identify the keyword using the indico keyword extraction api, while iteratively remove all words not critical to the underlying food item (Code may be found in [Recipe Generation RNN](/Scrips/Recipe-Generation-RNN)). My second approach, which was ultimately used to train the [Ingredent Identifier](/Scripts/Ingredient_identifier), was to start with a cleaned list of ingredients initially scraped from enchantedlearning.com

It is critical that image labels are as clean as possible, otherwise the neural network will have difficulty learning. It was also important to allow the model to have multi-word labels to represent items such as bell pepper. A useful extension of this project may be to vectorize the labels, so that the net will learn that the ingredient similarity, for example, misclassifying beef and steak is closer than chicken and peas. Vectorization methods require tokenization of text first. This may be an area worth explore further.

### Image Processing ([Code](/Scripts/Preprocessing))

Neural networks were trained with raw image data, and convolved imaged data that was passed though the 2014 image net winner, VGG-16 from Oxford. Transfer learning proved more fruitful given the limited size of my dataset. Activations were taken at the end of layer 30, before flattening to dense layers. It would be interesting to compare results using activations taken after these dense layers, but I did not have time to explore this comparison. Here is a rough illustration of how the image vectorization process:

<img src="figures/image_vectorization.png" width="">
<br><br>

Images were downsized to 100x100 so that I could iterate through training multiple models, in the time allotted for capstone projects.

### Neural Network Architecture ([Code](/Scripts/Ingredient_identifier/build_models.py))

My architecture went though multiple iterations, ultimately I settled on preprocessing images with VGG-16, and passing those activations into 3 hidden dense layers. My output layer consists of a sigmoid activation for each ingredient, and uses binary crossentropy loss.

# Results

This section provides some high level results for how my final net performed, and then some examples of how it predicts on a particular image.

### Overall results ([Code](/Scripts/Ingredient_identifier/evaluate_models.py))

__ChefNet:__ 48% Recall, 38% Precision

__ChefNet without VGG-16 Processing:__ 46% Recall, 35% Precision

__Naive simulation:__ 6% Recall, 21% Precision

The algorithm was able to perform better for some ingredients than other. Below you may see what classes had best recall (top 10 ranged from 75%-100%). The net had better recall for those ingredients that were more frequent in the dataset:

<img src="figures/recall_wordcloud.png" width="400">

Below represents the top classes in terms of Precision (top 10 range form 60%-100%), note that these classes are different. In general, the net was more precise with ingredients it only predicted a few times:

<img src="figures/precision_wordcloud.png" width="400">

### Example Predictions

Here are some examples of how well ChefNet predicted:

##### Carrot Cake

<img src="figures/Carrot_cake_slide.png" width="">

##### My Lunch last Monday

<img src="figures/Monday_lunch_slide.png" width="">

### Data Issues

The data is not perfect, below is a slide that shows two different images for the same recipe. Not only can the sugar cookie look completely different based on the decoration decisions, but there are also misplaced pictures.

As an example, below you may see two user submitted images for the same sugar cookie recipe. The first illustrates how a cooking may look like just about anything, the second shows how users misplace images:

<img src="figures/User_imgs.png" width="">

# Try ChefNet yourself

First you will need to install these dependencies, in addition to Conda:

* [Keras](http://keras.io/)

* [Skimage](http://scikit-image.org/)

* [HDF5](http://docs.h5py.org/en/latest/build.html)

You will need to download the weights of my trained convolutional neural net and place the .h5 in the [Models](/models) folder

My weights: [CNN Weights](https://drive.google.com/file/d/0B53_Ht6DdCsGMy1GTDkwR0piODg/view?usp=sharing)

You will also need to download the weights of trained VGG-16 and place the .h5 file in the [VGG Weights](/vgg_weights) folder.

VGG-16 weights: [vgg16_weights.h5](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view)

Next you should move any image file you would like to predict on into the [images](/images) folder.

Now you can run [predict_ingredients.py](/Scripts/Ingredient_identifier) to have the model make your predictions (make sure you navigate to the [ingredient identifier](/Scripts/Ingredient_identifier) before running ipython).

If you run the script in ipython, you may just run `predict_user_photo(model, vocab)` to predict additional photos without reloading the model.

Here is an example of how it should look:

<img src="figures/demo.png" width="">

# Next Steps

There are a number of next steps that can be taken with this project.

* The model may benefit from further tuning, and more neural network structures could be explored. It may also benefit from training on full size images.

* Additional data may be scraped from other recipe websites to create a larger dataset.

* Another extension may involve true image captioning at a character or word level. I started exploring this option, but found that is was less useful toward my motivation of predicting underlying ingredients.

### Recipe Generation with Recurrent Neural Networks

Separate from the ingredient identifier, I've also developed a model that will generate novel ingredients character by character, based on the cleaned recipes scraped from allrecipes.com. Here are a couple examples, I'm not sure how well they would turn out:

* banana, lemon juice, cream mushroom soup, milk, cheese sauce

* garlic, cheese, cheddar cheese, salt, cheese, sausage, garlic, pork, bacon hotme, chicken, sesame, asparagus, bread, cheese, bacon, hamburger bun

You may notice __bacon hotme__ is not an actual ingredient. This is an odd result that can come out of character by character text generation.

All scripts for this maybe found in [Recipe Generation RNN](Scripts/Recipe_Generation_RNN).

# Thank you

Big thank you to Jesse Lieman-Sifry for the inspiration behind this project, as well as to my Galvanize Instructors and peers for the continuous help and suggestions along the way. It was a pleasure to work with all of you.

### References

* VGG Net Representation:

``` Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman
arXiv:1409.1556 ```
