'''
Text Classifier: An algorithm that is able to classify a text based on it's contents. 
Classic example, an email as spam or not - applies to two distinct things. 

Sentiment Analysis - positive or negative connotation. 
'''

import nltk 
import random 
from nltk.tokenize import word_tokenize

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
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

short_pos = open("short_reviews/newpositive.txt", "r").read()
short_neg = open("short_reviews/newnegative.txt", "r").read()

documents = []
for r in short_pos.split('\n'):
	documents.append((r,"pos"))
for r in short_neg.split('\n'):
	documents.append((r,"neg"))

all_words = []
for w in word_tokenize(short_pos):
	all_words.append(w.lower())
for w in word_tokenize(short_neg): 
	all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000] # a list of (most of) the words


def find_features(document): 
	''' Function returns a dictionary of all the words in the top 5000 and if they're
		in the document (passed in) '''
	words = word_tokenize(document)
	features = {}
	for w in word_features: 
		features[w] = (w in words)
	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

training_set = featuresets[:10000] 
testing_set = featuresets[10000:] 

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algorithm accuracy %:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(10)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("Multinomial Naive Bayes Algorithm accuracy %:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("Gaussian Naive Bayes Algorithm accuracy %:", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("Bernoulli Naive Bayes Algorithm accuracy %:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Algorithm accuracy %:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Algorithm accuracy %:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC Algorithm accuracy %:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Algorithm accuracy %:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Algorithm accuracy %:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(classifier, MNB_classifier, BNB_classifier, LogisticRegression_classifier, 
								  SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)
print("VoteClassifier Algorithm accuracy %:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
