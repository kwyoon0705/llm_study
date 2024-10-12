from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import random

# KoBART 모델 및 토크나이저 불러오기 / Load KoBART model and tokenizer
model_name = "gogamza/kobart-summarization"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


# 치킨집 주문 처리 챗봇 / Chicken Shop Order Processing Chatbot
def chicken_order_chatbot(user_input):
    # 사용자가 요청한 주문 정보 처리 / Process user's order request
    if "가격" in user_input or "비용" in user_input or "얼마" in user_input:
        return check_price(user_input)
    elif any(word in user_input for word in ["메뉴", "추천", "치킨 종류"]):
        return recommend_menu()
    elif any(word in user_input for word in ["주문", "시키다", "배달"]):
        return process_order(user_input)
    elif any(word in user_input for word in ["배달", "배송", "배달 가능"]):
        return "네, 배달이 가능합니다. 주문하실 치킨 종류를 말씀해 주세요."
    else:
        # KoBART 모델을 사용하여 응답 생성 / Generate response using KoBART for other questions
        return generate_response_with_kobart(user_input)


# 메뉴 추천 기능 / Recommend Menu
def recommend_menu():
    menu = ["후라이드 치킨", "양념 치킨", "반반 치킨", "간장 치킨", "허니 갈릭 치킨"]
    return f"저희 추천 메뉴는 {random.choice(menu)}입니다. 어떤 치킨을 주문하시겠어요?"


# 가격 확인 기능 / Check Price
def check_price(order_text):
    # 가격 정보 딕셔너리 / Price information dictionary
    prices = {
        "후라이드": 15000,
        "양념": 16000,
        "반반": 16000,
        "간장": 17000,
        "허니 갈릭": 18000
    }

    for chicken, price in prices.items():
        if chicken in order_text:
            # 주문 개수 파악 / Detect the number of orders
            count = 1  # 기본 1마리로 가정 / Default to 1 chicken
            words = order_text.split()
            for word in words:
                if "마리" in word:
                    try:
                        count = int(word.replace("마리", ""))
                    except ValueError:
                        pass

            total_price = price * count
            return f"{chicken} 치킨 {count}마리의 가격은 {total_price:,}원입니다."

    return "주문 가능한 치킨의 종류는 후라이드, 양념, 반반, 간장, 허니 갈릭입니다."


# 주문 처리 기능 / Process Order
def process_order(order_text):
    chicken_types = ["후라이드", "양념", "반반", "간장", "허니 갈릭"]
    for chicken in chicken_types:
        if chicken in order_text:
            return f"{chicken} 치킨을 주문하셨습니다. 배달 주소와 전화번호를 알려주세요."
    return "주문하실 치킨 종류를 다시 말씀해 주세요."


# KoBART를 사용하여 응답 생성 / Generate response using KoBART
def generate_response_with_kobart(user_input):
    # 입력 텍스트 토크나이징 / Tokenize input text
    inputs = tokenizer(user_input, return_tensors="pt", max_length=1024, truncation=True)

    # 응답 생성 / Generate response
    response_ids = model.generate(
        inputs['input_ids'],
        max_length=100,
        num_beams=4,
        repetition_penalty=100.0,  # 반복 방지 벌점 설정 / Set repetition penalty
        length_penalty=1.0,
        no_repeat_ngram_size=3,  # 3-그램 반복 방지 설정 / Set n-gram repeat prevention
        early_stopping=True,
        temperature=0.7  # 온도 설정 / Set temperature for diversity in generation
    )

    # 생성된 응답 디코딩 / Decode generated response
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text


# 챗봇 실행 / Run Chatbot
if __name__ == "__main__":
    print("안녕하세요! 치킨 주문을 도와드리겠습니다. 질문이 있으시면 말씀해주세요.")
    while True:
        user_input = input("사용자: ")
        if user_input.lower() in ["종료", "끝", "그만"]:
            print("챗봇: 이용해 주셔서 감사합니다. 좋은 하루 되세요!")
            break

        response = chicken_order_chatbot(user_input)
        print("챗봇:", response)
