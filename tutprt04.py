# Part of Speech (POS) Tagging is labeling the pos to every word in the text using tuples. 
# The function can be applied after we've split by word.


from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer # PunktSentenceTokenizer is a ML tokenizer!

# get text
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

process_content()

def process_content(): 
	try:
		for i in tokenized: 
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)

			print(tagged)

	except Exception, e:
		print(str(e))
