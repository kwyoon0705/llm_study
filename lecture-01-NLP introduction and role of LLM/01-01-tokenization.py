import nltk
import numpy as np

# Ensure the punkt data is downloaded
nltk.download('punkt_tab')

# Tokenization
text = "NLP is fascinating."
tokens = nltk.word_tokenize(text)
print("Tokens:", tokens)  # Output: ['NLP', 'is', 'fascinating', '.']

# One-hot encoding representation
vocab = sorted(set(tokens))
print("Vocabulary:", vocab)  # ['.', 'NLP', 'fascinating', 'is']

one_hot_vectors = np.eye(len(vocab))[np.array([vocab.index(token) for token in tokens])]

# Display the one-hot vectors for each token
for token, vector in zip(tokens, one_hot_vectors):
    print(f"Token: {token} -> Vector: {vector}")