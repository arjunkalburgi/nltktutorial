# Text Classifier: An algorithm that is able to classify a text based on it's contents. 
# Classic example, an email as spam or not - applies to two distinct things. 

# Sentiment Analysis - positive or negative connotation. 

import nltk 
import random 
from nltk.corpus import movie_reviews

# create a tuple - where the first element is a list of the words in the text
# and the second element is the sentiment rating (pos/neg)
documents = [(list(movie_reviews.words(fileid)), category) 
						 for category in movie_reviews.categories()
						 for fileid in movie_reviews.fileid(category)]

# shuffle the docs so that it's not bias
random.shuffle(documents)

''' Now we can see the review and the sentiment analysis of that rating! '''

# create a list of all the words used in these movie reviews. 
# so that we can see what are the most commonly used words and then
# 
all_words = []
for w in movie_reviews.words(): 
	all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15)) # most of these are stopwords...
print(all_words["stupid"]) # how many times does "stupid" come up?
