import openai


news_article = """
오늘 주식 시장은 주요 기술 기업들이 분기별 실적 보고서를 발표하면서 큰 변동을 보였다.
애플은 아이폰 판매와 서비스 부문의 성장에 힘입어 예상보다 높은 수익 증가를 보고했다.
반면 구글의 모회사인 알파벳은 광고 수익이 감소하여 주가가 하락했다.
투자자들은 이러한 상황이 기술 분야의 광범위한 추세를 나타낼 수 있다는 점에서 주목하고 있다.
전체 시장은 혼조세를 보였으며, 일부 지수는 상승한 반면 다른 지수는 하락하며 거래를 마감했다.
"""
# OpenAI API 키 설정
openai.api_key = "your-openai-api-key-here"

# 요약 생성 함수
def summarize_article(article):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"다음 뉴스 기사를 요약해 주세요:\\n\\n{article}"}
        ],
        max_tokens=150,
        temperature=0.5
    )
    summary = response.choices[0].message['content'].strip()
    return summary

summary = summarize_article(news_article)
print("Summary:", summary)