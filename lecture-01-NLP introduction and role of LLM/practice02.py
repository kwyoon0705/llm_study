import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag


news_article = """
오늘 주식 시장은 주요 기술 기업들이 분기별 실적 보고서를 발표하면서 큰 변동을 보였다.
애플은 아이폰 판매와 서비스 부문의 성장에 힘입어 예상보다 높은 수익 증가를 보고했다.
반면 구글의 모회사인 알파벳은 광고 수익이 감소하여 주가가 하락했다.
투자자들은 이러한 상황이 기술 분야의 광범위한 추세를 나타낼 수 있다는 점에서 주목하고 있다.
전체 시장은 혼조세를 보였으며, 일부 지수는 상승한 반면 다른 지수는 하락하며 거래를 마감했다.
"""
nltk.download('averaged_perceptron_tagger')

# 단어 토큰화
words = word_tokenize(news_article)

# 품사 태깅
pos_tags = pos_tag(words)
print("POS Tags:", pos_tags)

# 주어(명사)와 동사만 출력
nouns_and_verbs = [word for word, pos in pos_tags if pos.startswith('N') or pos.startswith('V')]
print("Nouns and Verbs:", nouns_and_verbs)