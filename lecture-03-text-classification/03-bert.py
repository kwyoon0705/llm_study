# 1. 라이브러리 임포트
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import openai
import re

# 2. 영화 리뷰 데이터 생성 및 데이터프레임 만들기 (데이터 수 증가)
# data = {
#     "review_id": range(1, 101),
#     # "review": [
#     #               "이 영화는 정말 환상적이었어요! 스토리가 매우 흡입력이 있었습니다.",
#     #               "영화의 모든 순간이 싫었어요. 연기가 형편없었어요.",
#     #               "비주얼은 멋졌지만, 줄거리가 부족했어요.",
#     #               "즐거운 관람이었고, 멋진 배우들의 연기였습니다.",
#     #               "제 취향이 아니었어요, 너무 느리고 지루했어요.",
#     #               "정말 감동적인 이야기였고, 눈물이 났어요.",
#     #               "대사가 너무 뻔했어요. 예측 가능한 전개였어요.",
#     #               "음악이 정말 좋았고, 분위기를 잘 살렸어요.",
#     #               "기대 이하였습니다. 더 잘 만들 수 있었을 것 같아요.",
#     #               "배우들의 케미가 좋았고, 재미있게 봤어요.",
#     #               "액션 장면이 너무 많아서 지루했어요.",
#     #               "스토리가 탄탄하고 캐릭터가 매력적이었어요.",
#     #               "너무 어두운 내용이라서 기분이 안 좋았어요.",
#     #               "코미디 요소가 많아서 웃으면서 봤습니다.",
#     #               "전반적으로 지루하고 감정선이 약했어요.",
#     #               "화면이 아름답고, 연출이 훌륭했어요.",
#     #               "캐릭터들이 매력적이지 않아서 몰입이 안 됐어요.",
#     #               "끝까지 긴장감이 넘쳐서 손에 땀을 쥐게 했어요.",
#     #               "스토리가 엉성하고, 끝이 허무했어요.",
#     #               "다시 보고 싶을 만큼 좋은 영화였습니다."
#     #           ] * 5,  # 데이터 수를 늘리기 위해 5배 반복
#     "review": [
#                   "선생님이 친절하고 설명도 쉽게 해주셔서 이해하기 쉬웠어요. 많은 도움이 되었습니다.",
#                   "강의 자료가 잘 준비되어 있었고, 선생님이 질문에 친절하게 답해주셨어요.",
#                   "수업 내용이 체계적이고 실습 위주로 진행되어서 매우 유익했습니다.",
#                   "강의실 환경이 쾌적하고 수업 시간이 잘 지켜져서 좋았습니다.",
#                   "강사님이 열정적으로 가르쳐주셔서 동기부여가 많이 되었습니다.",
#                   "친절한 상담과 꼼꼼한 관리로 학습 계획을 잘 세울 수 있었습니다.",
#                   "수업 시간이 너무 길어서 집중하기 힘들었습니다. 조금 더 짧았으면 좋겠습니다.",
#                   "진도가 너무 빠르게 나가서 따라가기가 힘들었습니다.",
#                   "학습 자료가 부족해서 아쉬웠습니다. 더 많은 자료가 제공되면 좋겠습니다.",
#                   "실습 시간이 부족해서 이론만 배우는 느낌이 들었습니다.",
#                   "선생님이 학생 개개인에게 신경을 많이 써주셔서 좋았습니다.",
#                   "강의 내용이 너무 어려워서 이해하기 힘들었습니다. 기초부터 다시 설명해주시면 좋겠습니다.",
#                   "수업이 재미있고 유익해서 시간이 금방 갔습니다.",
#                   "학생들 간의 토론 시간을 많이 가져서 서로 배우는 점이 많았습니다.",
#                   "실제 사례를 많이 들어주셔서 이해하는 데 큰 도움이 되었습니다.",
#                   "온라인 강의도 병행해주셔서 시간 활용이 좋았습니다.",
#                   "수업 내용이 일관되지 않고 산만해서 집중하기 힘들었습니다.",
#                   "선생님의 발음이 정확하지 않아서 듣기 힘들었습니다.",
#                   "학습 목표가 명확하게 제시되어 있어서 좋았습니다.",
#                   "복습 자료가 잘 준비되어 있어서 학습한 내용을 다시 정리하기 좋았습니다."
#               ] * 5,  # 데이터 수를 늘리기 위해 5배 반복
#     # "sentiment": [
#     #                  "긍정", "부정", "부정", "긍정", "부정", "긍정", "부정", "긍정", "부정", "긍정",
#     #                  "부정", "긍정", "부정", "긍정", "부정", "긍정", "부정", "긍정", "부정", "긍정"
#     #              ] * 5  # 데이터 수를 늘리기 위해 5배 반복
#     "sentiment": [
#                      "긍정", "긍정", "긍정", "긍정", "긍정", "긍정", "부정", "부정", "부정", "부정",
#                      "긍정", "부정", "긍정", "긍정", "긍정", "긍정", "부정", "부정", "긍정", "긍정"
#                  ] * 5  # 데이터 수를 늘리기 위해 5배 반복
# }
data = {
    "review_id": range(1, 161),
    "review": [
                  "죽고싶어요.",
                  "이거 너무 재밌네요.",
                  "집에 가고 싶어요.",
                  "ㅋㅋㅋ 이거 돌았다 ㅋㅋ",
                  "이대로만 하면 우승할듯 ",
                  "아 요즘 너무 못하네 진짜",
                  "우리 잘할 수 있어!!",
                  "지면 나도 자살한다 ㅅㄱ",
                  "퇴근 언제하냐고...",
                  "집가서 치킨 먹어야지",
                  "이번에 보너스 받았나 아주 나이스",
                  "아 우리 팀장 겁나 쎄게 때리고 싶다.",
                  "월급 올려줘!!!!",
                  "우리 회사 망하는거 아님?",
                  "이번에 시험 붙으면 바로 여행간다",
                  "이번에 여친 사귐 ㅎㅎ"
              ] * 10,
    "sentiment": [
                     "자살의심", "즐거움", "우울함", "즐거움", "기대감", "걱정", "기대감", "자살의심", "우울함", "기대감", "행복함",
                     "때리고 싶음", "화남", "걱정", "기대감", "행복함"
                 ] * 10
}

df = pd.DataFrame(data)

# 3. 학습 및 검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# 4. 레이블 인코딩
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 5. BERT 토크나이저 설정
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# 6. 커스텀 Dataset 클래스 정의
class MovieReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 7. Dataset과 DataLoader 설정
train_dataset = MovieReviewDataset(X_train.to_numpy(), y_train_encoded, tokenizer)
test_dataset = MovieReviewDataset(X_test.to_numpy(), y_test_encoded, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)

# 8. BERT 모델 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                      num_labels=len(label_encoder.classes_))
model = model.to(device)

# 9. 옵티마이저와 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# 10. 학습 루프
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)


# 11. 학습 실행
for epoch in range(3):
    print(f'Epoch {epoch + 1}/{3}')
    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        device,
        scheduler
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')


# 12. 검증 데이터 평가
def eval_model(model, data_loader, device):
    model = model.eval()
    reviews, predictions, true_labels = [], [], []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            reviews.extend(d["review_text"])
            predictions.extend(preds)
            true_labels.extend(labels)

    return reviews, predictions, true_labels


# 13. 예측 결과와 설명 생성
openai.api_key = ""


def generate_explanation(review, prediction):
    sentiment = label_encoder.inverse_transform([prediction])[0]
    # prompt = (
    #     f"다음 영화 리뷰의 감성은 '{sentiment}'입니다. 리뷰: \"{review}\". "
    #     "이 리뷰가 왜 이런 감성으로 분류되었는지 설명해 주세요."
    # )
    prompt = (
        f"다음 인터넷 댓글의 감성은 '{sentiment}'입니다. 리뷰: \"{review}\". "
        "이 리뷰가 왜 이런 감성으로 분류되었는지 설명해 주세요."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message['content']


# 14. 테스트 데이터에 대한 예측 및 설명 생성
reviews, predictions, true_labels = eval_model(model, test_loader, device)

for i in range(5):  # 일부 테스트 데이터에 대해 예제 실행
    review = reviews[i]
    prediction = predictions[i].cpu().item()
    explanation = generate_explanation(review, prediction)
    print(f"리뷰: {review}\n예측: {label_encoder.inverse_transform([prediction])[0]}\n설명: {explanation}\n")
