# Colab 환경에서 필요한 패키지 설치
# Colab에서 사용할 때는 런타임이 끊기면 설치된 내용이 지워지기 때문에 런타임을 새로시작할 때는 openai 모듈을 새로 설치해야 오류가 없습니다.

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import spacy
import openai

# NLTK 다운로드 (최초 실행 시)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# SpaCy 한국어 모델 로드
nlp = spacy.load("ko_core_news_sm")

# OpenAI API 키 설정
openai.api_key = ""

# 예제 한국어 뉴스 기사 텍스트
news_article = """
오늘 주식 시장은 주요 기술 기업들이 분기별 실적 보고서를 발표하면서 큰 변동을 보였다.
애플은 아이폰 판매와 서비스 부문의 성장에 힘입어 예상보다 높은 수익 증가를 보고했다.
반면 구글의 모회사인 알파벳은 광고 수익이 감소하여 주가가 하락했다.
투자자들은 이러한 상황이 기술 분야의 광범위한 추세를 나타낼 수 있다는 점에서 주목하고 있다.
전체 시장은 혼조세를 보였으며, 일부 지수는 상승한 반면 다른 지수는 하락하며 거래를 마감했다.
"""

# 1. 문장 토큰화
sentences = sent_tokenize(news_article)
print("Sentences:", sentences)

# 2. 단어 토큰화 및 품사 태깅 (영어 기반으로 설명을 보여주기 위해 예시를 유지)
words = word_tokenize(news_article)
pos_tags = pos_tag(words)
print("POS Tags:", pos_tags)

# 3. 개체명 인식(NER)
doc = nlp(news_article)
print("\nNamed Entities:")
for ent in doc.ents:
    print(ent.text, ent.label_)

# 4. 요약 생성
def summarize_article(article):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"다음 뉴스 기사를 요약해 주세요:\\n\\n{article}"}
        ],
        max_tokens=150,
        temperature=0.5
    )
    summary = response.choices[0].message['content'].strip()
    return summary

summary = summarize_article(news_article)
print("\nSummary:", summary)