from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import openai
import os

openai.api_key = ''

# M2M100 모델 및 토크나이저 불러오기 / Load M2M100 model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)


# 영어를 한국어로 번역하는 함수 / Function to translate English to Korean
def translate_english_to_korean(text):
    tokenizer.src_lang = "en"
    inputs = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("ko"))
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text


# LLM을 사용하여 번역의 자연스러움 개선 / Improve Translation with LLM
def improve_translation_with_llm(text):
    # LLM을 사용하여 자연스러운 번역 생성 / Generate a more natural translation using LLM
    messages = [
        {"role": "system",
         "content": "You are an assistant that helps improve translations to make them more natural and fluent."},
        {"role": "user",
         "content": f"Please improve the following Korean translation to make it more natural and fluent:\n\n{text}\n\nImproved Translation:"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1000,
        temperature=0.7,
        top_p=1.0,
        n=1
    )

    improved_text = response.choices[0].message['content'].strip()
    return improved_text


# 번역 및 개선 실행 / Execute Translation and Improvement
if __name__ == "__main__":
    # 입력 영어 문장 / Input English Sentence
    while True:
        english_text = input("Your Message: ")
        if english_text == "종료" or english_text.lower() == "quit":
            print("Good Bye.")
            break
        # 1. M2M100을 사용한 초기 번역 / Initial translation using M2M100
        initial_translation = translate_english_to_korean(english_text)
        print("Input English Sentence:", english_text)
        print("Initial Translation (Korean):", initial_translation)

        # 2. LLM을 사용하여 번역의 자연스러움 개선 / Improve translation using LLM
        improved_translation = improve_translation_with_llm(initial_translation)
        print("Improved Translation (Korean):", improved_translation)