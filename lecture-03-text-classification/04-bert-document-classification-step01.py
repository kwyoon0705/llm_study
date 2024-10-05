import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 문서 의도 예제 데이터 생성
data = {
    "document": [
        "어디에서 이 제품을 구입할 수 있나요?",
        "다음 주까지 이 작업을 완료하세요.",
        "이 문제를 어떻게 해결할 수 있나요?",
        "도와주셔서 정말 감사합니다.",
        "서비스가 형편없습니다. 다시 이용하지 않을 것입니다.",
        "이 제품은 어디서 찾을 수 있나요?",
        "이 기능의 사용 방법을 알려주세요.",
        "새로운 버전을 어디서 다운로드할 수 있나요?",
        "이 문제에 대한 자세한 정보가 필요합니다.",
        "이 제품의 사용 설명서를 어디서 찾을 수 있나요?",
        "이 일을 빨리 끝내 주세요.",
        "모든 보고서를 내일까지 제출하세요.",
        "이 작업을 완료하는 데 시간이 얼마나 걸리나요?",
        "지금 바로 이 작업을 시작하세요.",
        "이 요청을 지체 없이 처리해 주세요.",
        "이 오류를 어떻게 고칠 수 있나요?",
        "이 기능은 어떻게 활성화하나요?",
        "이 문제를 해결하는 다른 방법이 있나요?",
        "왜 이런 현상이 발생하는지 알려줄 수 있나요?",
        "이 상황에서 어떤 조치를 취해야 하나요?",
        "당신의 지원에 깊이 감사드립니다.",
        "문제를 신속히 해결해 주셔서 감사합니다.",
        "귀하의 도움이 큰 도움이 되었습니다.",
        "적극적인 지원에 대해 감사드립니다.",
        "이 서비스는 매우 실망스럽습니다.",
        "제품 품질이 기대 이하입니다.",
        "고객 지원이 너무 느립니다.",
        "이 서비스는 전혀 만족스럽지 않습니다.",
        "다시는 이 서비스를 이용하지 않을 것입니다.",
        "배송이 너무 늦어 불만입니다.",
        "이 서비스의 가격은 얼마인가요?",
        "어떤 기능이 제공되나요?",
        "이 제품의 사양을 알려주세요.",
        "이 제품은 어떤 용도로 사용되나요?",
        "이 문제를 해결할 수 있는 사람은 누구인가요?",
        "이 기능에 대해 자세히 설명해 주세요.",
        "이 문제의 원인을 찾는 방법은 무엇인가요?",
        "이 항목에 대한 세부 정보가 필요합니다.",
        "지원해 주셔서 감사합니다.",
        "귀하의 도움에 진심으로 감사드립니다.",
        "여러분의 노고에 감사드립니다.",
        "서비스가 개선되어 정말 기쁩니다.",
        "고객 서비스가 너무 불친절합니다.",
        "제품의 품질이 너무 떨어집니다.",
        "이 문제로 인해 큰 불편을 겪었습니다.",
        "서비스 이용에 매우 실망했습니다.",
        "내일까지 이 작업을 끝내세요.",
        "즉시 이 문제를 해결하세요.",
        "이 일을 가능한 한 빨리 처리해 주세요.",
        "이 요청을 신속히 이행해 주세요.",
        "왜 이런 오류가 발생하는지 설명해 주세요.",
        "이 문제를 해결하기 위한 단계를 알려 주세요.",
        "이 기능을 사용할 수 없는 이유는 무엇인가요?",
        "어떤 방식으로 접근해야 할까요?",
        "정말 친절하게 대해 주셔서 감사합니다.",
        "빠른 처리에 감사드립니다.",
        "문제를 해결해 주셔서 감사합니다.",
        "도움을 주셔서 매우 기쁩니다.",
        "서비스가 너무 느립니다. 개선이 필요합니다.",
        "배송이 너무 지연됩니다.",
        "제품의 품질이 기대 이하입니다.",
        "고객 지원이 만족스럽지 않습니다.",
        "이 제품은 왜 이렇게 비싼가요?",
        "어떤 보증이 제공되나요?",
        "이 서비스는 어떻게 이용하나요?",
        "다른 대안은 무엇인가요?",
        "어떻게 하면 이 문제를 더 빨리 해결할 수 있을까요?",
        "서비스 품질이 정말 훌륭합니다. 고맙습니다.",
        "고객 서비스가 매우 친절했습니다.",
        "제품이 기대 이상으로 좋았습니다.",
        "이 기능을 추가해 주셔서 감사합니다.",
        "이용해 주셔서 감사합니다.",
        "서비스가 정말 좋았습니다.",
        "이 문제를 더 이상 겪고 싶지 않습니다.",
        "다시는 이 제품을 구매하지 않을 것입니다.",
        "너무 실망스럽습니다.",
        "제품이 제 기대에 미치지 못했습니다.",
        "이 문제에 대해 불만을 제기하고 싶습니다.",
        "어떤 방법으로 환불을 받을 수 있나요?",
        "환불 절차를 안내해 주세요.",
        "서비스를 개선해 주셔서 감사합니다.",
        "이 문제를 해결해 주셔서 기쁩니다.",
        "다른 질문이 있습니다. 답변해 주세요.",
        "추가 정보가 필요합니다.",
        "이 상황에서 어떻게 해야 하나요?",
        "다음 단계를 안내해 주세요.",
        "이 제품을 추천해 주시겠어요?",
        "서비스 이용에 대해 추가 질문이 있습니다.",
        "문제가 여전히 해결되지 않았습니다.",
        "이 문제를 해결하는 데 얼마나 걸릴까요?",
        "어떻게 하면 이 상황을 개선할 수 있을까요?",
        "이 기능에 대해 더 알고 싶습니다."
    ],
    "intent": [
        "정보 요청", "명령", "질문", "감사", "불만",
        "정보 요청", "정보 요청", "명령", "질문", "정보 요청",
        "명령", "명령", "질문", "명령", "명령",
        "질문", "질문", "질문", "질문", "질문",
        "감사", "감사", "감사", "감사", "불만",
        "불만", "불만", "불만", "불만", "불만",
        "정보 요청", "정보 요청", "정보 요청", "정보 요청", "질문",
        "질문", "정보 요청", "정보 요청", "감사", "감사",
        "감사", "감사", "불만", "불만", "불만",
        "불만", "명령", "명령", "명령", "명령",
        "질문", "질문", "질문", "질문", "감사",
        "감사", "감사", "감사", "불만", "불만",
        "불만", "불만", "정보 요청", "정보 요청", "정보 요청",
        "정보 요청", "질문", "질문", "질문", "질문",
        "감사", "감사", "감사", "감사", "불만",
        "불만", "불만", "불만", "불만", "불만",
        "불만", "불만", "불만", "불만", "불만",
        "정보 요청", "정보 요청", "질문", "질문", "질문",
        "질문", "정보 요청", "질문", "질문", "질문"
    ]
}

# 2. 레이블 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(data['intent'])

# 3. BERT 토크나이저 설정
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# 4. 커스텀 Dataset 클래스 정의
class IntentDataset(Dataset):
    def __init__(self, documents, labels, tokenizer, max_len=64):
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, item):
        document = str(self.documents[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            document,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'document_text': document,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 5. Dataset과 DataLoader 설정
dataset = IntentDataset(data['document'], labels_encoded, tokenizer)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 6. BERT 모델 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                      num_labels=len(label_encoder.classes_))
model = model.to(device)

# 7. 옵티마이저 및 학습 설정
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

# 8. 모델 학습
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # 역전파 및 최적화
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 9. 모델 저장
model_save_path = "bert_intent_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"모델이 저장되었습니다: {model_save_path}")

# 10. 모델 평가
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 정확도 계산
accuracy = accuracy_score(all_labels, all_predictions)
print(f"모델 정확도: {accuracy:.4f}")
