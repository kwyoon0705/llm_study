from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 예시 문서들
documents = [
    TaggedDocument(words=["이", "영화", "정말", "재미있어요"], tags=["doc1"]),
    TaggedDocument(words=["이", "책은", "정말", "감동적이었어요"], tags=["doc2"]),
    TaggedDocument(words=["저는", "음악을", "좋아합니다"], tags=["doc3"])
]

# Doc2Vec 모델 학습
model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)

# 'doc1' 문서의 벡터 값 출력
print(model.dv['doc1'])

# 'doc1'과 유사한 문서 찾기
print(model.dv.most_similar('doc1'))