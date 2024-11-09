# 필요한 라이브러리 임포트
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.decomposition import PCA
import faiss
import numpy as np  # numpy를 추가로 임포트
from sentence_transformers import SentenceTransformer

# 특정 경로에 있는 폰트를 사용하도록 설정
font_path = 'D:/develop_study/artificial_intelligence/llm_study/lecture-06-vertor-database/content/NanumBarunGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)

# Matplotlib에서 폰트 설정
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

# 텍스트 데이터 준비
texts = [
    "강아지는 충성심이 강하다.",
    "고양이는 독립적인 성격을 가지고 있다.",
    "애완동물은 스트레스를 줄여준다.",
    "새로운 강아지를 입양했다.",
    "고양이의 털 빠짐을 관리하는 법.",
    "애완동물의 건강을 위한 정기적인 검진이 필요하다.",
    "반려동물과 산책은 좋은 운동이 된다.",
    "애완동물에게 적절한 사료를 선택하는 법.",
    "강아지 훈련은 인내심이 필요하다.",
    "고양이는 깨끗한 화장실을 좋아한다.",
    "애완동물과의 놀이 시간은 중요하다.",
    "강아지의 사회화 교육 방법.",
    "고양이에게 적합한 장난감 추천.",
    "반려동물의 정서적 안정을 위한 환경 만들기.",
    "애완동물의 털 관리 요령.",
    "강아지와 고양이의 식사 습관 차이.",
    "애완동물의 안전을 위한 집안 환경 조성.",
    "강아지의 건강한 간식을 만드는 법.",
    "고양이의 행동 문제 해결 방법.",
    "애완동물과 함께 여행할 때 유의할 점.",
    "강아지는 놀이를 통해 에너지를 발산한다.",
    "고양이는 높은 곳에 올라가는 것을 좋아한다.",
    "애완동물의 눈 건강을 지키는 방법.",
    "새로운 고양이 장난감을 구매했다.",
    "강아지의 발톱을 자르는 방법.",
    "고양이의 긁기 행동을 막는 방법.",
    "반려동물과의 유대감을 높이는 방법.",
    "애완동물의 피부 건강을 위한 관리법.",
    "강아지의 체중 관리를 위한 팁.",
    "고양이의 입양 절차와 준비사항.",
    "반려동물의 치아 건강을 유지하는 법.",
    "강아지와의 여행을 계획하는 법.",
    "고양이의 식단에 대한 고려사항.",
    "애완동물의 스트레스 신호를 알아차리는 방법.",
    "강아지의 사회적 상호작용을 돕는 법.",
    "고양이의 숨기기 행동 이해하기.",
    "반려동물의 올바른 목욕 방법.",
    "애완동물의 분리 불안을 다루는 법.",
    "강아지의 훈련 성공 사례 공유.",
    "고양이의 특이한 행동 이해하기.",
    "애완동물의 휴식 공간 꾸미기.",
    "강아지와 고양이의 상호작용 촉진 방법.",
    "고양이의 털 빠짐을 줄이는 방법.",
    "반려동물의 건강한 식단 구성.",
    "애완동물의 운동량을 늘리는 방법.",
    "강아지의 특수 간식 만들기.",
    "고양이와 강아지의 놀이 방식 차이.",
    "애완동물의 감정 표현 이해하기.",
    "강아지의 물건 씹기 행동 교정하기.",
    "고양이의 긁기 행동 대처법.",
    "반려동물의 안전을 위한 가정 내 조치.",
    "애완동물의 다양한 장난감 소개.",
    "강아지의 건강한 털 관리 팁.",
    "고양이의 장난감 선택 기준.",
    "애완동물과의 교감을 위한 놀이 방법."
]

# 텍스트 임베딩을 위한 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# FAISS 인덱스 생성 및 벡터 추가
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 예제 질의
query = "강아지 키우고 싶다."
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
