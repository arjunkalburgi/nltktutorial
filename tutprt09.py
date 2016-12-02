# The Corpus - nltk's database of body of text's
# We can access corpras in multiple ways

# Way 1: 

from nltk.corpus import gutenberg 
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

print(tok[:10])


# didn't teach more ways... awk