# Lemmatizing - similar to stemming except better. 

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better", pos="a")) # a for adjective 

