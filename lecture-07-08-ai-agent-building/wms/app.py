from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
import re

app = Flask(__name__)

# OpenAI API 키 설정
openai.api_key = ""

# data.txt 파일에서 데이터 불러오기
with open("data.txt", "r", encoding="utf-8") as file:
    data = file.readlines()

# 데이터프레임 생성
df = pd.DataFrame(data, columns=["description"])

# 날짜, 타입, 수량, 제품 컬럼 추출
df['date'] = pd.to_datetime(df['description'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])
df['type'] = df['description'].str.extract(r'(입고|재고|출고)')[0]
df['quantity'] = df['description'].str.extract(r'수량: (\d+)').fillna(0).astype(int)
df['product'] = df['description'].str.extract(r'서울우유 (\S+)')[0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']

    # ChatGPT를 사용하여 응답 생성
    initial_response = generate_response(user_query)

    # ChatGPT 응답을 분석하여 월별 데이터 추출 요청
    extracted_data_response = extract_monthly_data(initial_response)

    # 추출된 데이터를 그래프에 사용할 수 있는 형식으로 변환
    labels, quantities = parse_extracted_data(extracted_data_response)

    return jsonify(response=initial_response, labels=labels, quantities=quantities)


def generate_response(query):
    # 전체 데이터 요약 생성
    combined_data = "".join(df['description'])

    # LLM을 통해 요약된 데이터를 바탕으로 응답 생성
    prompt = f"chatGPT 너는 물류전문가야, 전체 WMS 데이터가 {combined_data}일 때 이 데이터의 내용으로 {query}에 대한 답변을 해줘"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response['choices'][0]['message']['content']


def extract_monthly_data(response_text):
    # ChatGPT에게 월별 데이터를 추출 요청
    prompt = f"다음 텍스트에서 월별 데이터(YYYY-MM: 수량)를 추출해줘:\n{response_text}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    return response['choices'][0]['message']['content']


def parse_extracted_data(extracted_data):
    # 추출된 데이터를 그래프용 데이터로 변환
    pattern = r'(\d{4}-\d{2}): (\d+)'  # YYYY-MM 형식으로 월별 데이터 추출
    response_data = re.findall(pattern, extracted_data)

    # 추출된 데이터를 딕셔너리 형태로 변환
    response_data_dict = {pd.Period(year_month, freq='M'): int(quantity) for year_month, quantity in response_data}

    # 전체 월 범위를 설정하고 데이터가 없는 월은 0으로 채우기
    all_months = pd.date_range(df['date'].min(), df['date'].max(), freq='MS').to_period('M')
    monthly_data = pd.Series(response_data_dict).reindex(all_months, fill_value=0)

    labels = monthly_data.index.astype(str).tolist()
    quantities = monthly_data.tolist()
    return labels, quantities


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)