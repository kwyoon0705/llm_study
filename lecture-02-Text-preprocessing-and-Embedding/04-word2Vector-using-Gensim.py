from gensim.models import Word2Vec

# 예시 문장들
sentences = [
    ["이", "영화", "정말", "재미있어요"],
    ["이", "책은", "정말", "감동적이었어요"],
    ["저는", "음악을", "좋아합니다"]
]

# Word2Vec 모델 학습
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# '영화'라는 단어의 벡터 값 출력
print(model.wv['영화'])

# '영화'와 유사한 단어 찾기
print(model.wv.most_similar('영화'))