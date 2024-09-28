from sklearn.feature_extraction.text import TfidfVectorizer

# 예제 문서 집합
documents = [
    "자연어 처리는 인공지능의 한 분야입니다.",
    "자연어 처리와 컴퓨터 비전은 인공지능의 두 가지 중요한 분야입니다.",
    "컴퓨터 비전은 이미지와 비디오 데이터를 처리하는 기술입니다."
]

# TF-IDF 벡터라이저 초기화
vectorizer = TfidfVectorizer()

# 문서에 TF-IDF 적용
tfidf_matrix = vectorizer.fit_transform(documents)

# 단어 리스트 출력
print("Vocabulary:", vectorizer.get_feature_names_out())

# TF-IDF 행렬 출력
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

# 특정 단어의 TF-IDF 값 출력 (예: '인공지능의')
word = '인공지능의'
word_index = vectorizer.vocabulary_.get(word)
if word_index is not None:
    print(f"\nTF-IDF value for '{word}':", tfidf_matrix[:, word_index].toarray())
else:
    print(f"'{word}' is not in the vocabulary.")