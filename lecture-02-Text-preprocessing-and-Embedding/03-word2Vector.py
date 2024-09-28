import numpy as np


# 코사인 유사도 계산 함수
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)  # 벡터의 내적
    norm_v1 = np.linalg.norm(v1)  # 벡터 v1의 크기
    norm_v2 = np.linalg.norm(v2)  # 벡터 v2의 크기
    return dot_product / (norm_v1 * norm_v2)  # 코사인 유사도 계산


# 예시 벡터
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 코사인 유사도 계산
similarity = cosine_similarity(v1, v2)
print(f"코사인 유사도: {similarity}")
