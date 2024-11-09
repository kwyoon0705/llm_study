# 필요한 라이브러리 임포트
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

openai.api_key = ""


def generate_answer_from_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']


# 제목과 조치사항을 포함한 딕셔너리 데이터 준비
responses = [
    {
        "title": "음주운전",
        "content": """
        도로교통법 제148조의2(벌칙) ① 제44조제1항 또는 제2항을 위반(자동차등 또는 노면전차를 운전한 경우로 한정한다. 다만, 개인형 이동장치를 운전한 경우는 제외한다. 이하 이 조에서 같다)하여 벌금 이상의 형을 선고받고 그 형이 확정된 날부터 10년 내에 다시 같은 조 제1항 또는 제2항을 위반한 사람(형이 실효된 사람도 포함한다)은 다음 각 호의 구분에 따라 처벌한다. <개정 2023. 1. 3.>
        1. 제44조제2항을 위반한 사람은 1년 이상 6년 이하의 징역이나 500만원 이상 3천만원 이하의 벌금에 처한다.
        2. 제44조제1항을 위반한 사람 중 혈중알코올농도가 0.2퍼센트 이상인 사람은 2년 이상 6년 이하의 징역이나 1천만원 이상 3천만원 이하의 벌금에 처한다.
        3. 제44조제1항을 위반한 사람 중 혈중알코올농도가 0.03퍼센트 이상 0.2퍼센트 미만인 사람은 1년 이상 5년 이하의 징역이나 500만원 이상 2천만원 이하의 벌금에 처한다.
        ② 술에 취한 상태에 있다고 인정할 만한 상당한 이유가 있는 사람으로서 제44조제2항에 따른 경찰공무원의 측정에 응하지 아니하는 사람(자동차등 또는 노면전차를 운전한 경우로 한정한다)은 1년 이상 5년 이하의 징역이나 500만원 이상 2천만원 이하의 벌금에 처한다. <개정 2023. 1. 3.>
        ③ 제44조제1항을 위반하여 술에 취한 상태에서 자동차등 또는 노면전차를 운전한 사람은 다음 각 호의 구분에 따라 처벌한다.
        1. 혈중알코올농도가 0.2퍼센트 이상인 사람은 2년 이상 5년 이하의 징역이나 1천만원 이상 2천만원 이하의 벌금
        2. 혈중알코올농도가 0.08퍼센트 이상 0.2퍼센트 미만인 사람은 1년 이상 2년 이하의 징역이나 500만원 이상 1천만원 이하의 벌금
        3. 혈중알코올농도가 0.03퍼센트 이상 0.08퍼센트 미만인 사람은 1년 이하의 징역이나 500만원 이하의 벌금
        ④ 제45조를 위반하여 약물로 인하여 정상적으로 운전하지 못할 우려가 있는 상태에서 자동차등 또는 노면전차를 운전한 사람은 3년 이하의 징역이나 1천만원 이하의 벌금에 처한다.
        """
    },
    {
        "title": "교통사고",
        "content": """
        도로교통법 제140조의2(새로운 조항): 이 조항은 특정 조건 하에서 발생하는 교통사고에 대한 규제를 명시하고 있으며, 교통사고의 중대성에 따라 차등 처벌을 규정합니다.
        주요 내용:
        1. 중대한 교통사고로 사상자가 발생한 경우, 사고 가해자는 3년 이상의 징역형에 처할 수 있습니다.
        2. 피해자의 중상해가 인정되는 경우, 1년 이상의 징역형 또는 벌금형에 처해질 수 있습니다.
        3. 가벼운 사고로 인한 벌금형 및 면허 정지 조치.
        앞차와 추돌하여 중상해가 발생.
        차량이 전복하여 중상해가 발생.
        """
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
query = "음주운전"
query_vector = model.encode([query])

# FAISS를 사용해 유사한 제목 검색 (Top 3)
k = 1  # 상위 3개 유사한 제목 출력
distances, indices = index.search(query_vector, k)

# 검색 결과 출력
print(f"질의: {query}")
print("유사한 A/S 응답:")
for idx, (dist, index) in enumerate(zip(distances[0], indices[0])):
    title = responses[index]["title"]
    action = responses[index]["content"]
    prompt = f"다음 도로교통법을 참고하여 아래 상황에 대한 법률적인 설명을 제공해주세요: {title}\n\n사고 상황: {action}"
    print(generate_answer_from_gpt(prompt))
    # print(f"{idx + 1}. 제목: {title}")
    # print(f"   조치사항: {action} (거리: {dist})")
