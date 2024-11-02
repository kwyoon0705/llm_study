from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, KEYWORD, DATETIME, ID
from whoosh.qparser import MultifieldParser, QueryParser, WildcardPlugin
from datetime import datetime
import os

# 1. 스키마 정의 (Schema Definition)
# Whoosh에서는 다양한 데이터 필드를 손쉽게 추가할 수 있으며, 추가적인 데이터 변경 시에도 별도의 스키마 마이그레이션이 필요하지 않음
schema = Schema(
    doc_id=ID(stored=True, unique=True),
    title=TEXT(stored=True),
    content=TEXT(stored=True),
    author=TEXT(stored=True),
    tags=KEYWORD(stored=True, commas=True, lowercase=True),
    published_date=DATETIME(stored=True)
)

# 2. 인덱스 저장할 디렉토리 생성
index_dir = "indexdir_flexible"
if not os.path.exists(index_dir):
    os.mkdir(index_dir)

# 3. 인덱스 생성
ix = create_in(index_dir, schema)
writer = ix.writer()

# 4. 문서 추가 (Ease of Adding Data)
# Whoosh에서는 서로 다른 구조의 문서도 문제없이 추가할 수 있어, 다양한 데이터를 유연하게 저장할 수 있음
documents = [
    {
        "doc_id": "1",
        "title": "파이썬 소개",
        "content": "파이썬은 웹 개발, 데이터 분석 등 다양한 분야에서 사용되는 인기 있는 프로그래밍 언어입니다.",
        "author": "홍길동",
        "tags": "파이썬, 프로그래밍, 개발",
        "published_date": datetime(2021, 5, 17)
    },
    {
        "doc_id": "2",
        "title": "고급 파이썬 프로그래밍",
        "content": "이 문서는 파이썬 프로그래밍의 고급 주제인 데코레이터, 제너레이터, 메타클래스 등을 다룹니다.",
        "author": "이영희",
        "tags": "파이썬, 고급, 프로그래밍",
        "published_date": datetime(2022, 8, 24)
    },
    {
        "doc_id": "3",
        "title": "파이썬을 이용한 데이터 과학",
        "content": "파이썬을 사용하여 데이터 과학을 수행하는 방법을 배우고, Pandas, NumPy 및 데이터 시각화에 대해 알아봅니다.",
        "author": "홍길동",
        "tags": "파이썬, 데이터 과학, 판다스, 넘파이",
        "published_date": datetime(2023, 3, 12)
    },
    {
        "doc_id": "4",
        "title": "웹 개발 기초",
        "content": "이 문서는 HTML, CSS, JavaScript를 포함한 웹 개발의 기초를 소개합니다.",
        "author": "이영희",
        "tags": "웹 개발, html, css, 자바스크립트",
        "published_date": datetime(2020, 11, 5)
    }
]

for doc in documents:
    writer.add_document(
        doc_id=doc["doc_id"],
        title=doc["title"],
        content=doc["content"],
        author=doc["author"],
        tags=doc["tags"],
        published_date=doc["published_date"]
    )

writer.commit()

# 5. 검색 함수 (Convenient Search Options)
# 복잡한 SQL 대신 간단한 쿼리 구문으로 다양한 필드를 한번에 검색 가능
def search_documents(query_str):
    ix = open_dir(index_dir)
    with ix.searcher() as searcher:
        parser = MultifieldParser(["title", "content", "tags"], schema=ix.schema)
        query = parser.parse(query_str)
        results = searcher.search(query)
        for result in results:
            print(f"제목: {result['title']}, 저자: {result['author']}, 태그: {result['tags']}")

# 6. 예제 검색
# 텍스트 기반의 자연스러운 검색이 가능하고, 부분 일치, 와일드카드 등의 편리한 검색 기능 지원
print("전체 텍스트 검색 예제:")
search_documents("파이썬")

print("\\n특정 키워드와 일치하는 내용 검색:")
search_documents("고급 OR 데이터 과학")

print("\\n태그를 기반으로 한 검색 (Whoosh는 키워드 기반 검색도 지원):")
search_documents("파이썬 AND 개발")