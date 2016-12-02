'''
Text Classifier: An algorithm that is able to classify a text based on it's contents. 
Classic example, an email as spam or not - applies to two distinct things. 

Sentiment Analysis - positive or negative connotation. 
'''

import nltk 
import random 
from nltk.tokenize import word_tokenize

import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
	"""docstring for VoteClassifier"""
	def __init__(self, *classifiers):
		self._classifiers = classifiers
	def classify(self, features):
		votes = []
		for c in self._classifiers: 
			v = c.classify(features)
			votes.append(v)
		return mode(votes)
	def confidence(self, features): 
		votes = []
		for c in self._classifiers: 
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf		

def find_features(document): 
	''' Function returns a dictionary of all the words in the top 5000 and if they're
		in the document (passed in) '''
	words = word_tokenize(document)
	features = {}
	for w in word_features: 
		features[w] = (w in words)
	return features

###############################################################################################
short_pos = open("short_reviews/newpositive.txt", "r").read()
short_neg = open("short_reviews/newnegative.txt", "r").read()

documents = []
for r in short_pos.split('\n'):
	documents.append((r,"pos"))
for r in short_neg.split('\n'):
	documents.append((r,"neg"))

# documents_pickle = open("pickled_algorithms/documents.pickle", "wb")
# pickle.dump(documents, documents_pickle)
# documents_pickle.close()


###############################################################################################
all_words_pickle = open("pickled_algorithms/all_words.pickle", "rb")
all_words = pickle.load(all_words_pickle)
all_words_pickle.close()
print("all_words added")

all_words = nltk.FreqDist(all_words)


###############################################################################################
word_features_pickle = open("pickled_algorithms/word_features.pickle", "rb")
word_features = pickle.load(word_features_pickle)
word_features_pickle.close()
print("word_features added")


###############################################################################################

featuresets_pickle = open("pickled_algorithms/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_pickle)
featuresets_pickle.close()
print("featuresets added")

random.shuffle(featuresets)

training_set = featuresets[:10000] 
testing_set =  featuresets[10000:] 


###############################################################################################
classifier_pickle = open("pickled_algorithms/classifier.pickle", "rb")
classifier = pickle.load(classifier_pickle)
classifier_pickle.close()
print("classifier added")

print("Original Naive Bayes Algorithm accuracy %:   ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)


###############################################################################################
MNB_classifier_pickle = open("pickled_algorithms/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(MNB_classifier_pickle)
MNB_classifier_pickle.close()
print("MNB_classifier added")

print("Multinomial Naive Bayes Algorithm accuracy %:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


###############################################################################################
BNB_classifier_pickle = open("pickled_algorithms/BNB_classifier.pickle", "rb")
BNB_classifier = pickle.load(BNB_classifier_pickle)
BNB_classifier_pickle.close()
print("BNB_classifier added")

print("Bernoulli Naive Bayes Algorithm accuracy %:  ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)


###############################################################################################
LogisticRegressin_pickle = open("pickled_algorithms/LogisticRegressin.pickle", "rb")
LogisticRegression_classifier = pickle.load(LogisticRegressin_pickle)
LogisticRegressin_pickle.close()
print("LogisticRegression_classifier added")

print("LogisticRegression Algorithm accuracy %:     ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


###############################################################################################
LinearSVC_classifier_pickle = open("pickled_algorithms/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifier_pickle)
LinearSVC_classifier_pickle.close()
print("LinearSVC_classifier added")

print("LinearSVC Algorithm accuracy %:              ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)


###############################################################################################

voted_classifier = VoteClassifier(classifier, LinearSVC_classifier, MNB_classifier, BNB_classifier, LogisticRegression_classifier)
print("VoteClassifier Algorithm accuracy %:         ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

