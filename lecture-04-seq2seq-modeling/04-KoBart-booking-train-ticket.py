import pandas as pd
import re
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# KoBART 모델 및 토크나이저 불러오기 / Load KoBART model and tokenizer
model_name = "gogamza/kobart-summarization"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 다양한 기차 및 거리별 요금 데이터 / Train types and fare data
train_data = {
    "출발지": ["서울", "서울", "서울", "대전", "대전", "부산", "부산", "광주"],
    "도착지": ["대전", "부산", "광주", "부산", "광주", "서울", "대전", "서울"],
    "거리(km)": [150, 400, 300, 250, 180, 400, 250, 300],
    "KTX 요금(원)": [20000, 50000, 35000, 30000, 25000, 50000, 30000, 35000],
    "무궁화호 요금(원)": [10000, 25000, 20000, 18000, 15000, 25000, 18000, 20000],
    "새마을호 요금(원)": [15000, 35000, 30000, 25000, 20000, 35000, 25000, 30000]
}
train_df = pd.DataFrame(train_data)


# 기차 요금 조회 기능 / Function to Get Train Fare
def get_train_fare(user_input):
    # 사용자의 입력에서 출발지, 도착지, 기차 종류를 추출 / Extract departure, arrival, and train type from user input
    pattern = r"(\w+)에서 (\w+)까지 (\w+) 요금"
    match = re.search(pattern, user_input)

    if match:
        departure = match.group(1)
        arrival = match.group(2)
        train_type = match.group(3).upper()  # 기차 종류를 대문자로 변환 / Convert train type to uppercase

        # 기차 종류에 맞는 열 이름 찾기 / Find the correct fare column based on train type
        fare_column = ""
        if train_type == "KTX":
            fare_column = "KTX 요금(원)"
        elif train_type == "무궁화호":
            fare_column = "무궁화호 요금(원)"
        elif train_type == "새마을호":
            fare_column = "새마을호 요금(원)"
        else:
            return "KTX, 무궁화호, 새마을호 중 하나를 선택해 주세요."

        # 데이터프레임에서 출발지, 도착지에 해당하는 요금을 찾기 / Find the fare in the dataframe
        result = train_df[(train_df['출발지'] == departure) & (train_df['도착지'] == arrival)]

        if not result.empty:
            distance = result.iloc[0]['거리(km)']
            fare = result.iloc[0][fare_column]
            return f"{departure}에서 {arrival}까지의 거리는 {distance}km이며, {train_type} 요금은 {fare:,}원입니다."
        else:
            return "해당 경로에 대한 요금 정보를 찾을 수 없습니다. 출발지와 도착지를 다시 확인해 주세요."
    else:
        return generate_response_with_kobart(user_input)


# KoBART를 사용하여 응답 생성 / Generate response using KoBART
def generate_response_with_kobart(user_input):
    # 입력 텍스트 토크나이징 / Tokenize input text
    inputs = tokenizer(user_input, return_tensors="pt", max_length=1024, truncation=True)

    # 응답 생성 / Generate response
    response_ids = model.generate(
        inputs['input_ids'],
        max_length=100,
        num_beams=4,
        repetition_penalty=3.0,  # 반복 방지 벌점 설정 / Set repetition penalty
        length_penalty=1.0,
        no_repeat_ngram_size=3,  # 3-그램 반복 방지 설정 / Set n-gram repeat prevention
        early_stopping=True,
        temperature=0.7  # 온도 설정 / Set temperature for diversity in generation
    )

    # 생성된 응답 디코딩 / Decode generated response
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text


# 챗봇 실행 / Run Chatbot
if __name__ == "__main__":
    print("안녕하세요! 기차 요금을 안내해 드리겠습니다. 출발지와 도착지, 기차 종류를 말씀해 주세요. 예: '서울에서 부산까지 KTX 요금'")
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["종료", "끝", "그만"]:
            print("챗봇: 이용해 주셔서 감사합니다. 좋은 하루 되세요!")
            break

        response = get_train_fare(user_input)
        print("챗봇:", response)