from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
word = "running"
stem = stemmer.stem(word)
print(stem)  # Output: 'run'