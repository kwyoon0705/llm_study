from transformers import pipeline

nlp = pipeline("ner")
text = "John works at OpenAI."
entities = nlp(text)
print(entities)  # Output: [{'word': 'John', 'score': 0.99, 'entity': 'PER'}, ...]