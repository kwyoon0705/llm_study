from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

input_text = "Natural language processing is fascinating."
translated = model.generate(**tokenizer(input_text, return_tensors="pt", padding=True))
translation = tokenizer.decode(translated[0], skip_special_tokens=True)
print(translation)  # Output: 'Die Verarbeitung natürlicher Sprache ist faszinierend.'