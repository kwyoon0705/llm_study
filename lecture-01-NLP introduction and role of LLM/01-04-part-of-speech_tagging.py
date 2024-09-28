import nltk
nltk.download('averaged_perceptron_tagger_eng')
sentence = "He is running."
pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence))
print(pos_tags)  # Output: [('He', 'PRP'), ('is', 'VBZ'), ('running', 'VBG')]