# 필요한 라이브러리 임포트
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 제목과 조치사항을 포함한 딕셔너리 데이터 준비
responses = [
    {
        "title": "제품의 전원이 켜지지 않습니다.",
        "action": "전원 어댑터와 케이블을 확인해 주세요. 문제가 지속되면 서비스 센터 방문을 권장합니다."
    },
    {
        "title": "화면에 줄이 생겼어요.",
        "action": "화면 케이블 연결 상태를 확인하고, 문제가 해결되지 않으면 제품을 점검 받아보세요."
    },
    {
        "title": "소리가 나오지 않아요.",
        "action": "음량 설정 및 음소거 여부를 확인하고, 이어폰을 연결해 문제를 확인해 주세요."
    },
    {
        "title": "배터리 충전이 제대로 되지 않습니다.",
        "action": "충전기와 케이블 상태를 확인하고, 다른 충전기를 사용해보세요. 문제가 지속될 경우 배터리 점검을 받아보거나 교체하세요."
    },
    {
        "title": "제품이 과열되는 것 같아요.",
        "action": "사용 환경을 확인하고, 환기가 잘 되는 곳에서 사용해 주세요. 과열이 계속되면 점검이 필요합니다."
    }

]

# 텍스트 임베딩을 위한 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')
titles = [response["title"] for response in responses]
embeddings = model.encode(titles)

# FAISS 인덱스 생성 및 벡터 추가
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 예제 질의
query = "배터리가 충전되지 않아요."
query_vector = model.encode([query])

# FAISS를 사용해 유사한 제목 검색 (Top 3)
k = 3  # 상위 3개 유사한 제목 출력
distances, indices = index.search(query_vector, k)

# 검색 결과 출력
print(f"질의: {query}")
print("유사한 A/S 응답:")
for idx, (dist, index) in enumerate(zip(distances[0], indices[0])):
    title = responses[index]["title"]
    action = responses[index]["action"]
    print(f"{idx + 1}. 제목: {title}")
    print(f"   조치사항: {action} (거리: {dist})")
