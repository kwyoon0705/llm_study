import wikipediaapi
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import os

# 위키피디아 API 설정 (User-Agent를 명시적으로 지정)
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
wiki_wiki = wikipediaapi.Wikipedia(
    language='ko',
    user_agent=user_agent  # 한국어 위키피디아 사용, User-Agent 추가
)


# 위키피디아 문서 가져오기
def get_wikipedia_page(title):
    page = wiki_wiki.page(title)
    if page.exists():
        print(f"Retrieved page for {title}")  # 디버깅 출력
        return page.title, page.text
    else:
        print(f"Page for {title} does not exist.")  # 디버깅 출력
        return None, None


# 스키마 정의 (title과 content로 구성)
schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

# 인덱스 저장할 디렉토리 생성
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

# 인덱스 생성
ix = create_in("indexdir", schema)
writer = ix.writer()

# 위키피디아 문서 제목 리스트
# wiki_titles = ["라부아지에", "뉴튼", "데카르트", "퓨리에", "퀴리"]
wiki_titles = ["페이커 (프로게이머)", "구마유시", "케리아", "오너 (프로게이머)", "제우스 (프로게이머)"]

# 문서 추가 (위키피디아 문서 가져와서 인덱싱)
for title in wiki_titles:
    wiki_title, wiki_content = get_wikipedia_page(title)
    if wiki_title and wiki_content:
        writer.add_document(title=wiki_title, content=wiki_content)
        print(f"Indexed: {wiki_title}")  # 디버깅 출력

writer.commit()


# 검색
def search_whoosh(query_str):
    ix = open_dir("indexdir")
    with ix.searcher() as searcher:
        # 타이틀 필드와 콘텐츠 필드에 대해 각각 검색
        title_parser = QueryParser("title", ix.schema)
        content_parser = QueryParser("content", ix.schema)

        title_query = title_parser.parse(query_str)
        content_query = content_parser.parse(query_str)

        title_results = searcher.search(title_query)
        content_results = searcher.search(content_query)

        if title_results or content_results:
            for result in title_results:
                print(f"Title match: {result['title']}\nContent: {result['content'][:10000]}...\n")
            for result in content_results:
                print(f"Content match: {result['title']}\nContent: {result['content'][:10000]}...\n")
        else:
            print("검색 결과가 없습니다.")


# 예시 검색 실행
search_whoosh("구마유시")
