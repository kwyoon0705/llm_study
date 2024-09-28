from konlpy.tag import Okt  # Open Korean Text (Twitter tokenizer)

# 한국어 텍스트 예시
text = "오늘은 날씨가 정말 좋고, 기분이 좋아요! 하지만 내일은 비가 온다고 하네요."

# 1. 토큰화 (Tokenization)
okt = Okt()
tokens = okt.morphs(text)
print("토큰화:", tokens)

# 2. 불용어 제거 (Removing Stopwords)
stopwords = ['은', '가', '이', '하', '고', '내', '은', '도', '요']  # 일반적인 한국어 불용어 리스트
tokens = [word for word in tokens if word not in stopwords]
print("불용어 제거 후:", tokens)

# 3. 소문자 변환 (Lowercasing)
# 한국어는 대소문자가 없으므로 해당 단계는 생략.

# 4. 특수 문자 제거 (Removing Special Characters)
import re
clean_text = re.sub(r'[^\\w\\s]', '', text)  # \\w는 알파벳, 숫자, _를 의미, \\s는 공백
print("특수 문자 제거 후:", clean_text)

# 5. 어간 추출 또는 표제어 추출 (Stemming or Lemmatization)
# 한국어 형태소 분석을 통해 어근만 추출
lemmas = okt.morphs(text, stem=True)
print("어간 추출 후:", lemmas)