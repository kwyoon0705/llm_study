from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# 예제 문서 집합
documents = [
    "자연어 처리는 인공지능의 한 분야입니다.",
    "자연어 처리와 컴퓨터 비전은 인공지능의 두 가지 중요한 분야입니다.",
    "컴퓨터 비전은 이미지와 비디오 데이터를 처리하는 기술입니다."
]

# 문서를 토큰화
tokenized_documents = [word_tokenize(doc) for doc in documents]

# BM25 모델 초기화
bm25 = BM25Okapi(tokenized_documents)

# 예제 질의
query = "인공지능 분야"

# 질의를 토큰화
tokenized_query = word_tokenize(query)

# 문서들의 BM25 점수 계산
scores = bm25.get_scores(tokenized_query)

# 각 문서에 대한 BM25 점수 출력
print("BM25 Scores:", scores)

# 가장 관련성 높은 문서 출력
best_doc_index = scores.argmax()
print("\\nMost relevant document:", documents[best_doc_index])