# 필수 라이브러리 설치
# !pip install transformers torch scipy

from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

ongoing = True

# 1. BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')


# 2. 문장 임베딩 함수 정의 (BERT의 마지막 레이어 평균값 사용)
def get_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # 문장의 임베딩을 추출 (last_hidden_state의 평균값을 사용)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    except Exception as e:
        print(f"임베딩 처리 중 오류 발생: {e}")
        return None


# 3. 질문 및 문맥 정의
# context = [
#     "코레일에서는 다양한 열차 승차권을 예매할 수 있습니다.",
#     "서울에서 부산까지 KTX 요금은 약 59,800원입니다.",
#     "서울에서 부산까지 KTX 운행시간은 2시간 40분이 소요됩니다.",
#     "서울에서 대전까지 KTX 운행시간은 1시간이 소요됩니다.",
#     "기차표는 레츠코레일 웹사이트에서 예매할 수 있습니다.",
#     "한산대첩의 명장은 이순신장군입니다.",
#     "한산대첩은 1592년 8월 14일 일어난 전쟁입니다.",
#     "한산도 대첩을 설명하면 1592년 8월 14일 통영 한산도 앞바다에서 조선군이 일본군을 크게 무찌른 해전입니다."
# ]
context = [
    "학원의 이름은 휴먼IT학원입니다.",
    "학원의 수강비는 국비지원으로 전액 무료입니다.",
    "학원의 지점은 서울 영등포, 경기도 수원, 충청남도 천안에 위치해 있습니다.",
    "학원의 등록 절차는 학원을 내방하면 도와드리겠습니다."
]


def qna_system():
    global ongoing
    # question = "한산도 대첩은 언제 일어난 전쟁?"
    question = input("안녕하세요. 휴먼IT학원입니다. 무엇을 도와드릴까요?\n>>")

    if question == "quit":
        ongoing = False
        return

    # 4. 질문 임베딩 계산
    question_embedding = get_embedding(question)

    # 예외 처리: 질문 임베딩이 없으면 종료
    if question_embedding is None:
        print("질문에 대한 임베딩을 생성할 수 없습니다.")
    else:
        # 5. 문맥 내에서 각 문장의 유사도 계산
        similarities = []
        for sentence in context:
            sentence_embedding = get_embedding(sentence)
            if sentence_embedding is not None:
                similarity = 1 - cosine(question_embedding, sentence_embedding)  # 코사인 유사도 계산
                similarities.append(similarity)
                print(f"문장: '{sentence}'의 유사도: {similarity}")
            else:
                similarities.append(-1)  # 임베딩 오류 시 유사도를 -1로 설정

        # 6. 가장 높은 유사도를 가진 문장을 답변으로 선택
        if max(similarities) == -1:
            print("모든 문장에 대한 유사도 계산에 실패하였습니다.")
        else:
            best_answer = context[similarities.index(max(similarities))]
            print(f"\n질문: {question}")
            print(f"답변: {best_answer}")
            qna_system()


if ongoing:
    qna_system()
else:
    print("감사합니다. 휴먼IT학원이었습니다.")
