from nltk.tokenize import word_tokenize, sent_tokenize

# 예제 한국어 뉴스 기사 텍스트
news_article = """
오늘 주식 시장은 주요 기술 기업들이 분기별 실적 보고서를 발표하면서 큰 변동을 보였다.
애플은 아이폰 판매와 서비스 부문의 성장에 힘입어 예상보다 높은 수익 증가를 보고했다.
반면 구글의 모회사인 알파벳은 광고 수익이 감소하여 주가가 하락했다.
투자자들은 이러한 상황이 기술 분야의 광범위한 추세를 나타낼 수 있다는 점에서 주목하고 있다.
전체 시장은 혼조세를 보였으며, 일부 지수는 상승한 반면 다른 지수는 하락하며 거래를 마감했다.
"""

# 문장 토큰화
sentences = sent_tokenize(news_article)
print("Sentences:", sentences)

# 단어 토큰화 및 각 문장의 단어 개수 출력
for i, sentence in enumerate(sentences):
    words = word_tokenize(sentence)
    print(f"Sentence {i+1}: {words}")
    print(f"Number of words in Sentence {i+1}: {len(words)}")