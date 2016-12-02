# To begin understanding a language, the text must be split into sentences - as each sentence is one idea
# And the sentences must be split into words - as each word gives the meaning to the sentence.

# definitions
# 	tokenizing - breaking down text by word/sentences
# 	corpra - body of text(s). ex: medical journals, English language 
# 	lexicon - words and their meanings	

# nltk.download(); 



import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello, Mr. Boss, how are you doing today? Nice day outside! The sky is bright, nigga!"

print(sent_tokenize(example_text))

print(word_tokenize(example_text))

print("/n/n") 

for i in word_tokenize(example_text): 
	print(i)