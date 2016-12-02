# Many words have the same root word, stemming applies algorithms to make 
# all the words of the text into their root for easier understanding 

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ["lift", "lifting", "lifted", "lifter", "ball", "baller", "balled", "balling", "ballsy", "balls"]

print([ps.stem(w) for w in example_words]) #works really well!

new_text = "Man, that lifter lifted so much! He's lifting more now, so ballsy. The baller has a lot of lift \
	 while balling out, he balled and balls like a boss"

print([ps.stem(w) for w in word_tokenize(new_text)])