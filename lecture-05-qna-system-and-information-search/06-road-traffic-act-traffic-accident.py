from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser, WildcardPlugin
import os
import openai

# OpenAI API 키 설정
openai.api_key = ""

# 1. 스키마 정의 (title, content)
schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

# 2. 인덱스 저장할 디렉토리 생성
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

# 3. 인덱스 생성
ix = create_in("indexdir", schema)
writer = ix.writer()

# 도로교통법 및 교통사고 관련 법률 추가
wiki_data = [
    {
        "title": "도로교통법 제148조의2",
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

# 데이터 인덱스에 추가
for entry in wiki_data:
    writer.add_document(title=entry["title"], content=entry["content"])
    print(f"Indexed: {entry['title']}")

writer.commit()


# BAC 계산 함수
def calculate_bac_by_shots(shots, weight_kg, gender, appetizer, hours_passed=0):
    alcohol_oz_per_shot = 0.34
    total_alcohol_oz = shots * alcohol_oz_per_shot
    absorption_modifier = 0.9 if appetizer else 1.0
    bac = (total_alcohol_oz * 5.14) / (
                weight_kg * 2.20462 * (0.73 if gender == 'male' else 0.66)) * absorption_modifier - 0.015 * hours_passed
    return round(bac, 4)


# 검색 기능 구현
def search_law(query):
    ix = open_dir("indexdir")
    with ix.searcher() as searcher:
        parser = QueryParser("content", ix.schema)
        parser.add_plugin(WildcardPlugin())
        q = parser.parse(f"{query}*")
        results = searcher.search(q)
        return [result['content'] for result in results] if results else []


# GPT-3.5 Turbo로 답변 생성
def generate_answer_from_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']


# 질문 자동 분류 및 답변 생성
def answer_question_with_user_input():
    try:
        # 질문 유형 먼저 선택
        question_type = input("질문 유형을 선택해주세요 (음주운전/교통사고): ").strip().lower()

        # 관련 정보 입력
        if question_type == "음주운전":
            name = input("사용자의 이름을 입력해주세요: ")
            shots = int(input("소주를 몇 잔 마셨습니까? "))
            weight_kg = float(input("체중을 입력해주세요 (kg): "))
            gender_input = input("성별을 입력해주세요 (남성/여성): ")
            appetizer = input("어떤 안주를 드셨나요? (없으면 '없음'으로 입력): ")
            gender = 'male' if gender_input == "남성" else 'female' if gender_input == "여성" else None

            if gender is None:
                return "올바른 성별을 입력해주세요 ('남성' 또는 '여성')."

            # 안주 유무 확인
            has_appetizer = appetizer.lower() != '없음'

            # 혈중알콜농도 계산
            bac = calculate_bac_by_shots(shots, weight_kg, gender, has_appetizer)
            keyword = "도로교통법 제148조의2"
            search_results = search_law(keyword)

            if not search_results:
                return "관련된 법률 정보를 찾을 수 없습니다."

            bac_info = f"{name}님, 계산된 혈중알코올농도 (BAC)는 {bac}% 입니다.\n"
            appetizer_info = f"소주 {shots}잔과 함께 '{appetizer}'를 드셨군요.\n"
            dictionary_info = search_results[0]  # Get the dictionary info first
            prompt = f"{dictionary_info}\n\n{bac_info}{appetizer_info}위 내용을 참고하여 질문에 답변해 주세요."
            return generate_answer_from_gpt(prompt)

        elif question_type == "교통사고":
            incident_details = input("교통사고 상황을 간단히 설명해주세요: ")
            keyword = "교통사고"
            search_results = search_law(keyword)

            if not search_results:
                return "관련된 법률 정보를Here's the continuation and completion of the entire code that you requested:"

            prompt = f"다음 교통사고처리특례법을 참고하여 아래 상황에 대한 법률적인 설명을 제공해주세요: {search_results[0]}\n\n사고 상황: {incident_details}"
            return generate_answer_from_gpt(prompt)

        else:
            return "올바른 질문 유형을 선택해주세요 ('음주운전' 또는 '교통사고')."

    except ValueError:
        return "입력한 값이 올바르지 않습니다. 소주 잔수와 체중을 숫자로 입력해주세요."


# 예시 사용
print(answer_question_with_user_input())