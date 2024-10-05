import torch
from transformers import BertTokenizer, BertForSequenceClassification
import openai
from sklearn.preprocessing import LabelEncoder

# 1. 레이블 인코딩 초기화 (이전 학습 시 사용한 것과 동일하게 설정)
label_encoder = LabelEncoder()
label_encoder.fit([
    "정보 요청", "명령", "질문", "감사", "불만"
])

# 2. 모델 및 토크나이저 로딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',
                                                      num_labels=len(label_encoder.classes_))
model.load_state_dict(torch.load("bert_intent_model.pth", map_location=device))
model = model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# 3. 예측 함수 정의
def predict_intent(model, tokenizer, text, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)

    return prediction.item()


# 4. GPT-3.5를 통한 설명 생성 함수 정의
openai.api_key = ""


def generate_explanation(document, prediction):
    intent = label_encoder.inverse_transform([prediction])[0]

    prompt = f"""
    다음 문서의 의도는 '{intent}'입니다. 문서: "{document}".
    이 문서가 왜 이런 의도로 분류되었는지 설명해 주세요.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message['content']


# 5. 테스트 문장에 대한 예측 및 설명 생성
test_sentences = [
    "이 제품에 대한 추가 정보를 제공해 주세요.",
    "내일까지 이 작업을 끝내세요.",
    "이 문제를 어떻게 해결할 수 있나요?",
    "도움을 주셔서 정말 감사합니다.",
    "서비스가 전혀 만족스럽지 않습니다."
]

for sentence in test_sentences:
    # 의도 예측
    prediction = predict_intent(model, tokenizer, sentence, device)

    # GPT-3.5를 통한 설명 생성
    explanation = generate_explanation(sentence, prediction)
    print(f"문서: {sentence}\n예측된 의도: {label_encoder.inverse_transform([prediction])[0]}\n설명: {explanation}\n")
