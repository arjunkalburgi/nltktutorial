'''
Text Classifier: An algorithm that is able to classify a text based on it's contents. 
Classic example, an email as spam or not - applies to two distinct things. 

Sentiment Analysis - positive or negative connotation. 
'''

import nltk 
import random 
from nltk.corpus import movie_reviews

# create a tuple - where the first element is a list of the words in the text
# and the second element is the sentiment rating (pos/neg)
documents = []
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		documents.append((list(movie_reviews.words(fileid)), category))

# shuffle the docs so that it's not bias
random.shuffle(documents)

''' Now we can see the review and the sentiment analysis of that rating! '''

# create a list of all the words used in these movie reviews. 
# so that we can see what are the most commonly used words and then
''' wise to keep out stopwords? ''' 
all_words = []
for w in movie_reviews.words(): 
	all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)

''' Now we can see the most common words: all_words.most_common(15) 
	And how many times a word appears: all_words["stupid"]'''


def find_features(document): 
	''' Function returns a dictionary of all the words in the top 3000 and if they're
		in the document (passed in) '''
	features = dict()
	word_features = list(all_words.keys())[:3000] # a list of (most of) the words
	words = set(document) # all the different words
	for w in word_features: 
		features[w] = (w in words)
	return features

# print((find_features(movie_reviews.words("neg/cv000_29416.txt"))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]


'''
Naive Bayes Algorithm
Classifies things as either positive or negative sentiment- part of nltk's library

training_set: 
The idea is that we are looking at all the words and analyzing that if a word appears more 
often in negative reviews then that words is negative, or the opposite for positive. 

testing_set: 
The idea here is that based on the words that the computer knows are negative or positive,
we ask the computer to tell us if the review is positive or negative. 
Then we can check if the computer was right or wrong.

'''

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Naive Bayes Algorithm accuracy %:", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)