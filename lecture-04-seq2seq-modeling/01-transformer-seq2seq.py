from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# M2M100 모델 및 토크나이저 불러오기 / Load M2M100 model and tokenizer
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)


# 영어 문장을 한국어로 번역하는 함수 / Function to Translate English to Korean
def translate_english_to_korean(text):
    tokenizer.src_lang = "en"  # 원본 언어를 영어로 설정 / Set source language to English
    inputs = tokenizer(text, return_tensors="pt")

    # 번역된 토큰 생성 / Generate translated tokens
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("ko"))

    # 번역된 텍스트 디코딩 / Decode translated text
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text


# 번역 실행 / Execute Translation
if __name__ == "__main__":
    english_text = "I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation."

    # 번역 결과 출력 / Output Translation
    translated_text = translate_english_to_korean(english_text)
    print("Input English Sentence:", english_text)
    print("Translated Korean Sentence:", translated_text)
