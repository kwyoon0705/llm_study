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

# 한국어 위키피디아 데이터를 title과 content로 나누어 인덱싱
wiki_data = [
    {"title": "파이썬", "content": "파이썬(Python)은 범용 프로그래밍 언어입니다..."},
    {"title": "GPT", "content": "GPT는 OpenAI에서 개발한 언어 모델로, 자연어 처리에 사용됩니다..."},
    {"title": "대한민국", "content": "대한민국은 동아시아에 위치한 민주공화국입니다..."},
    # 더 많은 데이터 추가 가능
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
    # 가장 간단한 방법으로 키워드를 추출하는 과정. 실제로는 자연어 처리로 개선 가능
    keywords = ["파이썬", "GPT", "대한민국"]
    for keyword in keywords:
        if keyword in question:
            return keyword
    return None


# 7. 검색 결과 기반 GPT-3.5 답변 생성
def answer_question_with_wikipedia(question):
    keyword = extract_keyword(question)
    if not keyword:
        return "질문에서 키워드를 추출할 수 없습니다."

    search_results = search_wikipedia(keyword)

    if not search_results:
        return "위키피디아에서 관련된 정보를 찾을 수 없습니다."

    # 검색된 결과를 기반으로 GPT-3.5에게 답변 요청
    prompt = f"다음 위키피디아 데이터를 바탕으로 질문에 답해주세요: {search_results[0]}\n\n질문: {question}"
    answer = generate_answer_from_gpt(prompt)
    return answer


# 8. 예시 사용
question = "GPT"
print(answer_question_with_wikipedia(question))
