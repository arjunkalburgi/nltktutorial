# WordNet - the largest corpus from nltk, used for synonyms / antonyms / context

from nltk.corpus import wordnet


syns = wordnet.synsets("program")
print("synset for 'program': " + syns[0].name())
print("lemma for 'program': " + syns[0].lemmas()[0].name())
print("definition: " + syns[0].definition())
print("examples: " + syns[0].examples()) 


print(" ")
print(" ")


synonyms = []
antonyms = []
for syn in wordnet.synsets("good"): 
	for l in syn.lemmas(): 
		synonyms.append(l.name())
		if l.antonyms(): 
			antonyms.append(l.antonyms()[0].name())
print("synonyms for 'good': " + set(synonyms))
print("antonyms for 'good': " + set(antonyms))


print(" ")
print(" ")


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print("Similarity % between ship and boat" + w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print("Similarity % between ship and car" + w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
print("Similarity % between ship and cat" + w1.wup_similarity(w2))
