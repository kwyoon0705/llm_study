from whoosh.index import create_in, open_dir  # open_dir 추가
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import os

# 스키마 정의 (title과 content로 구성)
schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

# 인덱스 저장할 디렉토리 생성
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

# 인덱스 생성
ix = create_in("indexdir", schema)
writer = ix.writer()

# 문서 추가 (내용을 복잡하게 작성)
writer.add_document(
    title="윤건우",
    content=(
        "그는 1995년 경기도 시흥에서 태어났다."
        "키는 182센티미터이며, 몸무게는 비밀이다."
    )
)
writer.add_document(
    title="Python",
    content=(
        "Python은 1991년 Guido van Rossum이 처음 발표한 범용 프로그래밍 언어입니다. "
        "이 언어는 코드 가독성을 중시하며, 동적 타이핑, 메모리 관리 기능을 포함한 "
        "높은 수준의 내장 기능을 제공합니다. 다양한 프로그래밍 패러다임을 지원하며, "
        "특히 객체 지향, 명령형, 함수형 프로그래밍을 사용할 수 있습니다. "
        "주요 라이브러리로는 NumPy, pandas, Flask, Django 등이 있으며, "
        "현재는 데이터 분석, 웹 개발, 인공지능, 머신러닝 등 다양한 분야에서 폭넓게 사용되고 있습니다."
    )
)

writer.add_document(
    title="Whoosh",
    content=(
        "Whoosh는 Python으로 작성된 경량 텍스트 검색 및 인덱싱 라이브러리입니다. "
        "이 라이브러리는 Lucene과 유사한 기능을 제공하며, 빠르고 유연한 검색을 가능하게 합니다. "
        "Whoosh는 SQL 데이터베이스나 외부 서버와의 통합이 필요하지 않아, 작고 독립적인 애플리케이션에 적합합니다. "
        "텍스트 문서를 인덱싱하여 역파일(Inverted Index) 형태로 저장하고, "
        "이후 사용자의 검색 질의에 따라 적절한 문서를 빠르게 찾아내는 것이 주요 기능입니다. "
        "Whoosh는 풀 텍스트 검색, 유연한 쿼리 기능, 그리고 스코어링을 통해 검색 결과를 정렬할 수 있습니다."
    )
)

writer.add_document(
    title="GPT",
    content=(
        "GPT(Generative Pre-trained Transformer)는 OpenAI에서 개발한 자연어 처리 모델입니다. "
        "이 모델은 대규모 텍스트 데이터를 학습하여, 새로운 문장을 생성하거나, "
        "주어진 질문에 대한 답변을 제공할 수 있습니다. GPT는 트랜스포머(Transformer) "
        "구조를 기반으로 하며, 특히 다중 헤드 셀프 어텐션(Multi-head Self-attention) 기법을 사용하여 "
        "효율적으로 문맥을 이해합니다. 최근 발표된 GPT-3와 GPT-4는 1750억 개 이상의 매개변수를 사용해 "
        "더욱 정교하고 복잡한 언어 모델링을 가능하게 합니다. 이러한 모델은 챗봇, 번역, 요약, 질의 응답 등 "
        "다양한 자연어 처리 응용에 활용됩니다."
    )
)

writer.commit()


# 검색
def search_whoosh(query_str):
    ix = open_dir("indexdir")  # open_dir 사용
    with ix.searcher() as searcher:
        query = QueryParser("content", ix.schema).parse(query_str)
        results = searcher.search(query)
        for result in results:
            print(f"Title: {result['title']}\nContent: {result['content']}\n")


# 예시 검색 실행
search_whoosh("트랜스포머")
