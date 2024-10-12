import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 예제 데이터 생성
data = {
    "reply": [
        "특전사 병 출신이라고 떠들어 대던 문재앙은 탈북민도 대한민국 국민임에도 불구하고 북한 어선 탈북민을 강제 북송시켰고 서해 공무원 피살 사건에 대해서도 어떠한 조치도 하지 않고 오히려 월북 시도로 몰아갔다...",
        "잘했습니다....역시 믿음직 스러운 정부입니다",
        "문재인이는 귀순한 탈북민들조차도 의사에 반하여 눈에 안대까지 씌워가며 강제로 다시 북송시키던데 윤석열이는 레바논에서 위험에 빠진 우리 국민들을 군 수송기까지 동원하여 귀국시키는구나!!! 정말로 비교된다!!!",
        "이런게 선진국인 자유대한민국이다.",
        "조용히 국민들의 안전 귀국을 위해 긴급하게 조치를 취했구나. 국민의 안전이 국가의 최우선 의무인데 든든하다.",
        "이게 나라다  윤석렬 대통령님 고맙습니다 대한민국이 진정한 선진국이 되었습니다 공산당만 척결하면 very good ~",
        "레바논이 중동 한가운데 있으면서도 기독교인이 40%넘는 독특한 국가임. 그래서 국회의원도 이슬람,기독교 국회의원을 동일한 수로 맞춰서 뽑는 독특한 룰이 있음. 인종도 아르메니아인을 포함한 순수 백인계 비율이 높음. 그래서 지금까지 아랍전체를 대표하는 최고 인기가수들이 레바논 사람이 많음(Fairuz, Nancy Ajram등등). 그래서,, 중동,아랍세계가 다 복잡한 내막을 갖고있지만 이 레바논도 역사를 보면 꼬이고 꼬인 복잡한 역사가 있음(관심있는 분들은 검색해보셈)",
        "나라가 승승장구하는구나",
        "저 아이들 사진을 보고도 처참한 욕질이 나오는 문재인 이재명 돈봉투당 조국이 지지자들 댓글은.. 어우.. 사람이 안죽어서 화가났어? 그래야 아몰랑 석열~ 울부짖는데? 음주운전이나 하고 다니던 범죄 전과4범 이재명 따위를 위해? 막 사람이 죽어나가야되니? 어우..",
        "세금한푼 안내는 애들아냐? 돈내라그래",
        "정부가 빨리 철수 명령을 내리면,,,,알아서 개인각자가 비행기타고 미리 한국으로 들어와야하는거아니야?,,,,왜,,혈세를 낭비하면서 까지,,,,수송하는데,,,,,앞으로,,,,,전쟁나는 나라는 정부가 알아서 빨리 통보하보하고,,,,개인각자가 철수하게끔만들어라,,,만약,,,,국민혈세내는 군 수송기타고오는 인간들은,,,,평소비행기값 두배는 받아라,,,,왜,,,,꼭 위험지역을 군수송기까지 대절해서,,,자국민을 수송하는데,,,,개인각자가 알아서 미리 들어오라고,,,,",
        "2찍 보수 댓글부대로 보이는 놈들 또 여기서 여론조작 장난치네. 문재인 때는 코로나19 때 교민 수송, 아프카니스탄 교민 수송도 했는데, 그거 싹 다 잊었나? 윤석열 정부 레바논 97명 구출했다는 걸 앞세워 2찍 댓글부대로 보이는 놈들은 또 오버질하고 주작질하네.",
        "이게 대한민국 입니다",
        "윤석렬 대통령 역쉬!!\n잘한다~~~~~~",
        "정부  잘 한다",
        "이런건 멋있다... 민주당이었으면 나몰라라 했을꺼야",
        "뭉죄앙은 평양기여가서 자국민 인질들  쌩까고 냉면처먹고 있었지.",
        "그런다고 김건희 의혹이 쑤욱 들어갈거라고 생각하나? 국민들 안전과 귀국은  정부로서 당연히 해야 하는 일인거야... 김건희 사태에 대한 과심을 해소하기엔 너무 진부하잖아..."
    ],
    "intents": [
        "냉소", "자부심", "비교", "자부심", "신뢰", "자부심", 
        "분석", "자부심", "비난", "비난", "비판", "비판", "자부심",
        "칭찬", "칭찬", "비교", "비교", "비난"
    ]
}

# 2. 레이블 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(data["intents"])

# 3. BERT Tokenizer 설정
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


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
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "document_text": document,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# 5. Dataset과 DataLoader 설정
dataset = IntentDataset(data["reply"], labels_encoded, bert_tokenizer)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 6. BERT 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
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

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

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