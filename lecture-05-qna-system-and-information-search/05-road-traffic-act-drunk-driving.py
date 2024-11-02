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

# 도로교통법 제148조의2 관련 내용 추가
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
    }
]

# 데이터 인덱스에 추가
for entry in wiki_data:
    writer.add_document(title=entry["title"], content=entry["content"])
    print(f"Indexed: {entry['title']}")  # 인덱스된 데이터를 확인하는 출력

writer.commit()


# 4. 검색 기능 구현
def search_wikipedia(query):
    ix = open_dir("indexdir")
    with ix.searcher() as searcher:
        parser = QueryParser("content", ix.schema)
        parser.add_plugin(WildcardPlugin())  # Wildcard를 허용하는 플러그인 추가
        q = parser.parse(f"{query}*")  # 부분 일치를 허용하기 위해 Wildcard 쿼리 사용
        results = searcher.search(q)
        if results:
            return [result['content'] for result in results]
        else:
            return []


# 5. GPT-3.5 Turbo로 답변 생성
def generate_answer_from_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']


# 6. 검색어에서 키워드 추출
def extract_keyword(question):
    # 도로교통법이나 처벌 관련 키워드를 포함한 질문을 처리
    if "도로교통법" in question or "혈중 알콜농도" in question or "처벌" in question:
        return "도로교통법 제148조의2"
    return None


# 7. 검색 결과 기반 GPT-3.5 답변 생성
def answer_question_with_wikipedia(question):
    keyword = extract_keyword(question)
    if not keyword:
        return "질문에서 키워드를 추출할 수 없습니다."

    search_results = search_wikipedia(keyword)

    if not search_results:
        return "도로교통법에서 관련된 정보를 찾을 수 없습니다."

    # 검색된 결과를 기반으로 GPT-3.5에게 답변 요청
    prompt = f"다음 도로교통법 제148조의2 음주운전 위반 데이터를 바탕으로 질문에 답해주세요: {search_results[0]}\n\n질문: {question}"
    answer = generate_answer_from_gpt(prompt)
    return answer


# 8. 예시 사용
question = "혈중 알콜농도가 0.08인 사람이 받는 처벌은?"
print(answer_question_with_wikipedia(question))
