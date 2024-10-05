from flask import Flask, render_template, request
import qna as qna

# Flask 앱 생성
app = Flask(__name__)


# 기본 페이지
@app.route('/')
def index():
    return render_template('index.html')


# 질문 처리
@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    answer = qna.find_most_similar_answer(user_question)
    print(user_question)
    return render_template('index.html', question=user_question, answer=answer)


# Flask 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
