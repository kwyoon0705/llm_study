import openai

# OpenAI API 키 설정
openai.api_key = ''  # 여기에 본인의 API 키를 입력하세요

# 특정 펜션 정보 데이터
pension_info = {
    "오션뷰 펜션": {
        "위치": "강릉",
        "가격(1박/원)": 150000,
        "예약 가능 여부": "가능",
        "BBQ 가능 여부": "가능",
        "식사 제공 여부": "제공 안 함",
        "전화번호": "010-1234-5678",
        "애완동물 동반 여부": "불가능",
        "방 갯수": 3,
        "편의시설": ["Wi-Fi", "주차", "TV", "냉장고", "에어컨"]
    }
}


# 펜션 정보 질문 처리 함수
def get_pension_response(pension_name, question):
    if pension_name in pension_info:
        pension_details = pension_info[pension_name]

        # 질문에 따라 적절한 정보를 제공
        prompt = f"{pension_name}에 대한 정보입니다: {pension_details}. 사용자 질문: {question}.\n답변:"

        # GPT-3.5 Turbo API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )

        return response['choices'][0]['message']['content'].strip()
    else:
        return "해당 펜션 정보를 찾을 수 없습니다."


# 챗봇 실행
if __name__ == "__main__":
    print("안녕하세요! 펜션에 대한 문의를 도와드리겠습니다.")
    pension_name = input("펜션 이름을 입력하세요: ")

    while True:
        # 사용자로부터 펜션 이름과 질문을 입력받음
        if pension_name.lower() in ["종료", "끝", "그만"]:
            print("챗봇: 이용해 주셔서 감사합니다. 좋은 하루 되세요!")
            break

        user_input = input("사용자 질문을 입력하세요: ")
        if user_input.lower() in ["종료", "끝", "그만"]:
            print("챗봇: 이용해 주셔서 감사합니다. 좋은 하루 되세요!")
            break

        response = get_pension_response(pension_name, user_input)
        print("챗봇:", response)
