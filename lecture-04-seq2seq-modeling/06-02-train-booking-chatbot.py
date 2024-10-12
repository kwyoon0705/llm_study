import openai

# OpenAI API 키 설정
openai.api_key = ''  # 여기에 본인의 API 키를 입력하세요

# 특정 펜션 정보 데이터
train_data = {
    "출발지": ["서울", "서울", "서울", "대전", "대전", "부산", "부산", "광주"],
    "도착지": ["대전", "부산", "광주", "부산", "광주", "서울", "대전", "서울"],
    "거리(km)": [150, 400, 300, 250, 180, 400, 250, 300],
    "KTX 요금(원)": [20000, 50000, 35000, 30000, 25000, 50000, 30000, 35000],
    "무궁화호 요금(원)": [10000, 25000, 20000, 18000, 15000, 25000, 18000, 20000],
    "새마을호 요금(원)": [15000, 35000, 30000, 25000, 20000, 35000, 25000, 30000]
}


# 펜션 정보 질문 처리 함수
def get_pension_response(question):
    # 질문에 따라 적절한 정보를 제공
    prompt = f"열차 표에 대한 정보입니다: {train_data}. 사용자 질문: {question}.\n답변:"

    # GPT-3.5 Turbo API 호출
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7
    )

    return response['choices'][0]['message']['content'].strip()


# 챗봇 실행
if __name__ == "__main__":
    print("안녕하세요! 기차표 문의에 대한 문의를 도와드리겠습니다.")

    while True:
        user_input = input("사용자 질문을 입력하세요: ")
        if user_input.lower() in ["종료", "끝", "그만"]:
            print("챗봇: 이용해 주셔서 감사합니다. 좋은 하루 되세요!")
            break

        response = get_pension_response(user_input)
        print("챗봇:", response)
