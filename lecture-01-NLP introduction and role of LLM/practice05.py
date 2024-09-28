import nltk
import spacy
import openai

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("ko_core_news_sm")
openai.api_key = "your-openai-api-key-here"

# 사용자 입력 뉴스 기사
news_article = input("요약할 한국어 뉴스 기사를 입력하세요:\\n")

# 문장 토큰화
sentences = nltk.sent_tokenize(news_article)
print("Sentences:", sentences)

# 단어 토큰화 및 품사 태깅
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    print("POS Tags:", pos_tags)

# 개체명 인식(NER)
doc = nlp(news_article)
for ent in doc.ents:
    print(ent.text, ent.label_)

# 요약 생성
summary = summarize_article(news_article)
print("Summary:", summary)