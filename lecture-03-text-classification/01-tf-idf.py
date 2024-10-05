# 2. 라이브러리 임포트
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import openai

# 3. 영화 리뷰 데이터 생성 및 데이터프레임 만들기
data = {
    "review_id": range(1, 21),
    # "review": [
    #     "이 영화는 정말 환상적이었어요! 스토리가 매우 흡입력이 있었습니다.",
    #     "영화의 모든 순간이 싫었어요. 연기가 형편없었어요.",
    #     "비주얼은 멋졌지만, 줄거리가 부족했어요.",
    #     "즐거운 관람이었고, 멋진 배우들의 연기였습니다.",
    #     "제 취향이 아니었어요, 너무 느리고 지루했어요.",
    #     "정말 감동적인 이야기였고, 눈물이 났어요.",
    #     "대사가 너무 뻔했어요. 예측 가능한 전개였어요.",
    #     "음악이 정말 좋았고, 분위기를 잘 살렸어요.",
    #     "기대 이하였습니다. 더 잘 만들 수 있었을 것 같아요.",
    #     "배우들의 케미가 좋았고, 재미있게 봤어요.",
    #     "액션 장면이 너무 많아서 지루했어요.",
    #     "스토리가 탄탄하고 캐릭터가 매력적이었어요.",
    #     "너무 어두운 내용이라서 기분이 안 좋았어요.",
    #     "코미디 요소가 많아서 웃으면서 봤습니다.",
    #     "전반적으로 지루하고 감정선이 약했어요.",
    #     "화면이 아름답고, 연출이 훌륭했어요.",
    #     "캐릭터들이 매력적이지 않아서 몰입이 안 됐어요.",
    #     "끝까지 긴장감이 넘쳐서 손에 땀을 쥐게 했어요.",
    #     "스토리가 엉성하고, 끝이 허무했어요.",
    #     "다시 보고 싶을 만큼 좋은 영화였습니다."
    # ],
    "review": [
        "선생님이 친절하고 설명도 쉽게 해주셔서 이해하기 쉬웠어요. 많은 도움이 되었습니다.",
        "강의 자료가 잘 준비되어 있었고, 선생님이 질문에 친절하게 답해주셨어요.",
        "수업 내용이 체계적이고 실습 위주로 진행되어서 매우 유익했습니다.",
        "강의실 환경이 쾌적하고 수업 시간이 잘 지켜져서 좋았습니다.",
        "강사님이 열정적으로 가르쳐주셔서 동기부여가 많이 되었습니다.",
        "친절한 상담과 꼼꼼한 관리로 학습 계획을 잘 세울 수 있었습니다.",
        "수업 시간이 너무 길어서 집중하기 힘들었습니다. 조금 더 짧았으면 좋겠습니다.",
        "진도가 너무 빠르게 나가서 따라가기가 힘들었습니다.",
        "학습 자료가 부족해서 아쉬웠습니다. 더 많은 자료가 제공되면 좋겠습니다.",
        "실습 시간이 부족해서 이론만 배우는 느낌이 들었습니다.",
        "선생님이 학생 개개인에게 신경을 많이 써주셔서 좋았습니다.",
        "강의 내용이 너무 어려워서 이해하기 힘들었습니다. 기초부터 다시 설명해주시면 좋겠습니다.",
        "수업이 재미있고 유익해서 시간이 금방 갔습니다.",
        "학생들 간의 토론 시간을 많이 가져서 서로 배우는 점이 많았습니다.",
        "실제 사례를 많이 들어주셔서 이해하는 데 큰 도움이 되었습니다.",
        "온라인 강의도 병행해주셔서 시간 활용이 좋았습니다.",
        "수업 내용이 일관되지 않고 산만해서 집중하기 힘들었습니다.",
        "선생님의 발음이 정확하지 않아서 듣기 힘들었습니다.",
        "학습 목표가 명확하게 제시되어 있어서 좋았습니다.",
        "복습 자료가 잘 준비되어 있어서 학습한 내용을 다시 정리하기 좋았습니다."
    ],
    # "sentiment": [
    #     "긍정", "부정", "혼합", "긍정", "부정", "긍정", "부정", "긍정", "부정", "긍정",
    #     "부정", "긍정", "부정", "긍정", "부정", "긍정", "부정", "긍정", "부정", "긍정"
    # ]
    "sentiment": [
                    "긍정", "긍정", "긍정", "긍정", "긍정", "긍정", "부정", "부정", "부정", "부정",
                    "긍정", "부정", "긍정", "긍정", "긍정", "긍정", "부정", "부정", "긍정", "긍정"
                  ]
}

df = pd.DataFrame(data)

# 4. 학습 및 검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# 5. 한국어 불용어 목록 정의 (예시로 일부 불용어 추가)
korean_stopwords = ["정말", "매우", "모든", "너무", "정도", "이", "그", "저", "에서", "그리고"]

# 6. 감성 분석 모델 정의 및 학습(TD-IDF)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=korean_stopwords)),
    ('classifier', MultinomialNB())
])

model.fit(X_train, y_train)

# 7. 검증 데이터에 대한 모델 평가
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 8. OpenAI API 설정
openai.api_key = ""


# 9. 리뷰에 대한 설명을 생성하는 함수 정의
def generate_explanation(review, prediction):
    prompt = (
        f"다음 영화 리뷰의 감성은 '{prediction}'입니다. 리뷰: \"{review}\". "
        "이 리뷰가 왜 이런 감성으로 분류되었는지 설명해 주세요."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message['content']


# 10. 예측 및 설명 생성
for review in X_test[:5]:  # 일부 테스트 데이터에 대해 예제 실행
    prediction = model.predict([review])[0]
    explanation = generate_explanation(review, prediction)
    print(f"리뷰: {review}\n예측: {prediction}\n설명: {explanation}\n")
