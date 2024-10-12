from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# KoBART 모델 및 토크나이저 불러오기 / Load KoBART model and tokenizer
model_name = "gogamza/kobart-summarization"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 한국어 텍스트를 요약하는 함수 / Function to Summarize Korean Text
def summarize_korean_text(text):
    # 입력 텍스트 토크나이징 / Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

    # 요약 생성 / Generate summary
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=100,               # 요약된 텍스트의 최대 길이를 줄임
        num_beams=4,                  # 빔 서치
        repetition_penalty=3.0,       # 반복 방지 벌점 설정
        length_penalty=1.5,           # 길이 패널티 설정
        no_repeat_ngram_size=3,       # 3-그램 반복 방지 설정
        early_stopping=True,
        temperature=0.7               # 온도 설정을 낮게 조절
    )

    # 요약된 텍스트 디코딩 / Decode summarized text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# 요약 실행 / Execute Summary
if __name__ == "__main__":
    korean_text = (
        "오늘 날씨가 정말 좋습니다. 저는 공원에서 산책을 하고 싶습니다. "
        "어제는 비가 많이 와서 공원에 가지 못했지만, 오늘은 맑은 하늘과 따뜻한 온도 덕분에 기분이 좋습니다."
    )

    # 요약 결과 출력 / Output Summary
    summarized_text = summarize_korean_text(korean_text)
    print("Input Korean Text:", korean_text)
    print("Summarized Korean Text:", summarized_text)
