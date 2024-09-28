import re  # 정규 표현식 모듈
from konlpy.tag import Okt  # Open Korean Text 형태소 분석기

# 한국어 형태소 분석기 초기화
okt = Okt()

# 예시 한국어 영화 리뷰 데이터
reviews = [
    "정말 대단한 영화였습니다. 연출, 연기, 음악 모두 완벽했어요. 다음 편이 너무 기대됩니다!",
    "이 영화는 정말 실망스러웠어요. 처음부터 끝까지 스토리가 지루하고, 배우들의 연기도 별로였어요.",
    "중반까지는 조금 지루했지만 후반부의 반전은 예상치 못했습니다. 그 부분이 아주 흥미로웠어요!",
    "오랜만에 눈물을 흘리게 만든 영화였습니다. 감정선이 정말 뛰어나고, 여운이 오래 남는 영화네요.",
    "솔직히 기대 이하였습니다. 예고편 보고 기대 많이 했는데, 막상 보니 특별한 점이 없었어요.",
    "가족과 함께 즐길 수 있는 아주 따뜻한 영화였습니다. 어린이들과 함께 보기에 딱 좋습니다!",
    "액션이 너무 화려하고 시원시원했습니다. 다만 스토리가 조금 빈약한 느낌이 없지 않아 아쉽네요.",
    "배우들의 케미가 정말 좋았습니다. 특히 주인공의 감정 연기가 돋보였어요. 꼭 추천합니다!",
    "처음엔 살짝 지루했지만, 점점 빠져들었습니다. 마지막 장면은 정말 인상적이었어요!",
    "이 영화는 제 인생 영화입니다. 두 번, 세 번 다시 보고 싶을 만큼 정말 최고였어요."
]

# 불용어 정의 (일반적인 한국어 불용어)
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']


# 전처리 함수 정의
def preprocess_review(review):
    # 1. 특수 문자 제거
    review = re.sub(r'[^가-힣\s]', '', review)  # 한글과 공백을 제외한 문자 모두 제거

    # 2. 토큰화 및 불용어 제거
    tokens = okt.morphs(review)  # 형태소 단위로 토큰화
    tokens = [word for word in tokens if word not in stopwords]  # 불용어 제거

    # 3. 어간 추출 (형태소 분석기를 이용해 어간 추출)
    stemmed_tokens = [okt.pos(token, stem=True) for token in tokens]

    # 결과 반환 (토큰화된 단어 리스트)
    return tokens


# 각 리뷰에 대해 전처리 수행
processed_reviews = [preprocess_review(review) for review in reviews]

# 결과 출력
for i, review in enumerate(processed_reviews):
    print(f"원본 리뷰 {i + 1}: {reviews[i]}")
    print(f"전처리된 리뷰 {i + 1}: {review}")
    print()