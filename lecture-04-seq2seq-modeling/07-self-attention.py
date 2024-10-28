import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from konlpy.tag import Okt
import torch
from transformers import BertModel, BertTokenizer

# 한글 폰트 설정
font_path = 'C:/Users/USER/AppData/Local/Microsoft/Windows/Fonts/NanumBarunGothic.ttf'  # 나눔바른고딕 폰트 파일 경로
plt.rcParams['font.family'] = 'NanumBarunGothic'  # 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# BERT 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# 입력 문장 토큰화 및 임베딩 벡터 생성
input_sentence = "안녕하세요, 오늘 날씨가 참 좋네요."
okt = Okt()
tokens = okt.morphs(input_sentence)

# BERT 토크나이저를 이용하여 토큰을 인덱스로 변환
inputs = tokenizer(tokens, return_tensors='pt', padding=True, is_split_into_words=True)

# BERT 모델을 통해 임베딩 생성
with torch.no_grad():
    outputs = model(**inputs)
    # 마지막 히든 스테이트에서 각 토큰의 임베딩 벡터를 추출
    embeddings = outputs.last_hidden_state[0]

# Query, Key, Value 벡터 생성 (BERT 임베딩 사용)
query = embeddings.numpy()
key = embeddings.numpy()
value = embeddings.numpy()


# Softmax 함수 정의
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


# Attention 함수 정의
def attention(query, key, value):
    score = np.dot(query,
                   key.T)  # Query와 Key 간의 유사도 계산, 여기서 모두 같은 문장이 입력되어 유사도를 비교하기 때문에 즉 자기 자신의 유사도를 측정하기 때문에 Self Attention이라고 한다.
    d_k = query.shape[-1]
    scaled_score = score / np.sqrt(d_k)  # 스케일링
    attention_weights = softmax(scaled_score)  # Softmax 정규화
    output = np.dot(attention_weights, value)  # 가중합 계산
    return output, attention_weights


# Attention 계산
output, weights = attention(query, key, value)

# Attention Weights 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(weights, annot=True, fmt=".2f", cmap='Blues', xticklabels=tokens, yticklabels=tokens, cbar=True)
plt.xlabel('Key Tokens')
plt.ylabel('Query Tokens')
plt.title('Attention Weights Heatmap')
plt.show()
