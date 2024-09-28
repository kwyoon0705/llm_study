from flask import Flask, render_template, request
import re
from konlpy.tag import Okt  # Open Korean Text 형태소 분석기
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Flask 앱 생성
app = Flask(__name__)

# 한국어 형태소 분석기 초기화
okt = Okt()

# 펜션 예약 관련 미리 정의된 답변 목록
answers = [
    "예약은 홈페이지에서 가능합니다. 날짜를 선택하고 결제를 진행해 주세요.",
    "전화로도 예약이 가능합니다. 연락처는 010-1234-5678입니다.",
    "취소는 예약일 기준 3일 전까지 가능합니다. 그 이후에는 취소 수수료가 발생합니다.",
    "체크인은 오후 3시부터 가능합니다. 체크아웃은 오전 11시입니다.",
    "애완동물 동반은 가능합니다. 다만 추가 요금이 발생할 수 있습니다.",
    "바베큐 시설은 추가 요금을 내고 이용할 수 있습니다. 예약 시 옵션을 선택해 주세요.",
    "추가 인원 요금은 1인당 10,000원이 부과됩니다.",
    "무료 주차 공간이 마련되어 있습니다.",
    "예약 변경은 예약일 7일 전까지 가능합니다."
]

# 불용어 정의
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']


# 전처리 함수 정의
def preprocess_text(text):
    text = re.sub(r'[^가-힣\\s]', '', text)  # 한글과 공백만 남기기
    tokens = okt.morphs(text)  # 형태소 분석
    tokens = [word for word in tokens if word not in stopwords]  # 불용어 제거
    return tokens


# 답변에 대해 전처리 수행
processed_answers = [preprocess_text(answer) for answer in answers]

# Word2Vec 모델 학습
word2vec_model = Word2Vec(sentences=processed_answers, vector_size=100, window=5, min_count=1, sg=0)


# 문장 벡터 계산
def get_sentence_vector(sentence_tokens, model):
    vectors = [model.wv[word] for word in sentence_tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# 질문에 대한 가장 유사한 답변 찾기
def find_most_similar_answer(question, threshold=0.3):
    processed_question = preprocess_text(question)
    question_vector = get_sentence_vector(processed_question, word2vec_model)
    answer_vectors = np.array([get_sentence_vector(answer, word2vec_model) for answer in processed_answers])
    cosine_similarities = cosine_similarity([question_vector], answer_vectors)
    most_similar_idx = np.argmax(cosine_similarities)
    highest_similarity = cosine_similarities[0][most_similar_idx]

    if highest_similarity < threshold:
        return f"적절한 답변을 찾을 수 없습니다. (유사도: {highest_similarity:.2f})"
    else:
        return answers[most_similar_idx]


# 기본 페이지
@app.route('/')
def index():
    return render_template('index.html')


# 질문 처리
@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    answer = find_most_similar_answer(user_question)
    print(user_question)
    return render_template('index.html', question=user_question, answer=answer)


# Flask 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
