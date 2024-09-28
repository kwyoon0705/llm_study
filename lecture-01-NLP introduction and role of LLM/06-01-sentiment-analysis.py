from transformers import pipeline

# 감정 분석 파이프라인 초기화
nlp = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 한국어 예제 텍스트
text = "저는 자연어 처리가 정말 좋아요!"

# 감정 분석 수행
sentiment = nlp(text)

# 결과 확인 및 출력
if sentiment[0]['label'] == '5 stars':
    print("Positive")
elif sentiment[0]['label'] == '4 stars':
    print("Positive")
elif sentiment[0]['label'] == '3 stars':
    print("Neutral")
elif sentiment[0]['label'] == '2 stars':
    print("Negative")
elif sentiment[0]['label'] == '1 star':
    print("Negative")
else:
    print("Unknown sentiment")


# 결과 출력
# print(sentiment)  # Output: [{'label': '5 stars', 'score': 0.8432925343513489}]