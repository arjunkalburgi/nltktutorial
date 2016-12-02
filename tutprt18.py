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

all_words_f = open("all_words.pickle", "rb")
all_words = pickle.load(all_words_f)
all_words_f.close()
print("all_words added")
all_words = nltk.FreqDist(all_words)

word_features_f = open("word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()
print("word_features added")


def find_features(document): 
	''' Function returns a dictionary of all the words in the top 5000 and if they're
		in the document (passed in) '''
	words = word_tokenize(document)
	features = {}
	for w in word_features: 
		features[w] = (w in words)
	return features


# featuresets = [(find_features(rev), category) for (rev, category) in documents]
# print(len(featuresets))

featuresets_pickle1 = open("featuresets1.pickle", "rb")
featuresets1 = pickle.load(featuresets_pickle1)
featuresets_pickle1.close()

featuresets_pickle2 = open("featuresets2.pickle", "rb")
featuresets2 = pickle.load(featuresets_pickle2)
featuresets_pickle2.close()

featuresets_pickle3 = open("featuresets3.pickle", "rb")
featuresets3 = pickle.load(featuresets_pickle3)
featuresets_pickle3.close()

featuresets_pickle4 = open("featuresets4.pickle", "rb")
featuresets4 = pickle.load(featuresets_pickle4)
featuresets_pickle4.close()

featuresets = featuresets1 + featuresets2 + featuresets3 + featuresets4
print(len(featuresets))
random.shuffle(featuresets)

training_set = featuresets[:10000] 
testing_set =  featuresets[10000:] 

classifier_f = open("classifier.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
print("classifier added")
print("Original Naive Bayes Algorithm accuracy %:   ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

MNB_classifier_pickle = open("MNB_classifier.pickle", "wb")
pickle.dump(MNB_classifier, MNB_classifier_pickle)
MNB_classifier_pickle.close()
print("MNB_classifier_pickle done")

print("Multinomial Naive Bayes Algorithm accuracy %:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)

BNB_classifier_pickle = open("BNB_classifier.pickle", "wb")
pickle.dump(BNB_classifier, BNB_classifier_pickle)
BNB_classifier_pickle.close()
print("BNB_classifier_pickle done")

print("Bernoulli Naive Bayes Algorithm accuracy %:  ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

LogisticRegressin_pickle = open("LogisticRegressin.pickle", "wb")
pickle.dump(LogisticRegressin, LogisticRegressin_pickle)
LogisticRegressin_pickle.close()
print("LogisticRegressin_pickle done")

print("LogisticRegression Algorithm accuracy %:     ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)

MNB_classifier_pickle = open("MNB_classifier.pickle", "wb")
pickle.dump(MNB_classifier, MNB_classifier_pickle)
MNB_classifier_pickle.close()
print("MNB_classifier_pickle done")

print("SGDClassifier Algorithm accuracy %:          ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC Algorithm accuracy %:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

MNB_classifier_pickle = open("MNB_classifier.pickle", "wb")
pickle.dump(MNB_classifier, MNB_classifier_pickle)
MNB_classifier_pickle.close()
print("MNB_classifier_pickle done")

print("LinearSVC Algorithm accuracy %:              ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

MNB_classifier_pickle = open("MNB_classifier.pickle", "wb")
pickle.dump(MNB_classifier, MNB_classifier_pickle)
MNB_classifier_pickle.close()
print("MNB_classifier_pickle done")

print("NuSVC Algorithm accuracy %:                  ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(classifier, BNB_classifier, LogisticRegression_classifier, 
								  SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)
print("VoteClassifier Algorithm accuracy %:         ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)




# classifier_pickle = open("classifier.pickle", "wb")
# pickle.dump(classifier, classifier_pickle)
# classifier_pickle.close()

