import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.decomposition import PCA
import faiss
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# 특정 경로에 있는 폰트를 사용하도록 설정
font_path = 'D:/develop_study/artificial_intelligence/llm_study/lecture-06-vertor-database/content/NanumBarunGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)

# Matplotlib에서 폰트 설정
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# 디렉터리에서 모든 PDF 파일 텍스트를 추출하는 함수
def extract_texts_from_pdfs(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            doc = fitz.open(pdf_path)
            pdf_text = ""
            for page in doc:
                pdf_text += page.get_text()
            texts.append(pdf_text)
    return texts

# PDF 파일들이 있는 디렉터리 경로 설정
directory_path = "D:/develop_study/artificial_intelligence/llm_study/lecture-06-vertor-database/content/pdfs/"  # PDF 파일들이 있는 디렉터리 경로
pdf_texts = extract_texts_from_pdfs(directory_path)

# PDF에서 추출된 텍스트를 문장 단위로 나누기 (예시: 줄바꿈 기준으로 분리)
texts = []
for text in pdf_texts:
    sentences = text.split("\n")
    cleaned_sentences = [s.strip() for s in sentences if s.strip()]  # 빈 문장 제거
    texts.extend(cleaned_sentences)

# 텍스트 임베딩을 위한 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# FAISS 인덱스 생성 및 벡터 추가
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 예제 질의
query = "전북연구원 보수규정"
query_vector = model.encode([query])

# FAISS를 사용해 유사한 텍스트 검색 (Top 3)
k = 3  # 상위 3개 유사한 텍스트 출력
distances, indices = index.search(query_vector, k)

# 검색 결과 출력
print(f"질의: {query}")
print("유사한 텍스트:")
for idx, (dist, index) in enumerate(zip(distances[0], indices[0])):
    print(f"{idx + 1}. {texts[index]} (거리: {dist})")

# 기존 텍스트 + 질의 임베딩 추가
texts_with_query = texts + [query]
embeddings_with_query = np.vstack([embeddings, query_vector])

# PCA를 사용해 2D로 차원 축소
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_with_query)

# 그래프 그리기
plt.figure(figsize=(8, 6))
for i, text in enumerate(texts_with_query):
    color = 'red' if i == len(texts) else 'blue'  # 질의는 빨간색, 나머지는 파란색
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color)
    plt.text(embeddings_2d[i, 0] + 0.01, embeddings_2d[i, 1] + 0.01, text, fontsize=9, fontproperties=fontprop)

# 질의 텍스트와 유사한 텍스트를 연결하여 선으로 표시
for idx in indices[0]:
    plt.plot(
        [embeddings_2d[-1, 0], embeddings_2d[idx, 0]],
        [embeddings_2d[-1, 1], embeddings_2d[idx, 1]],
        'k--', alpha=0.5  # 검은 점선으로 연결
    )

plt.title("텍스트 벡터의 2D 시각화 (질의와 유사한 텍스트)", fontproperties=fontprop)
plt.xlabel("PCA Component 1", fontproperties=fontprop)
plt.ylabel("PCA Component 2", fontproperties=fontprop)
plt.grid(True)
plt.show()