from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import openai
import torch
import re

# 1. KoGPT-2 모델 로드 ("skt/kogpt2-base-v2")
kogpt2_model_name = "skt/kogpt2-base-v2"  # KoGPT-2 모델

# 모델과 토크나이저 로드 (BART, T5는 요약성능이 심하게 나쁨, KoGPT-2 역시 할루시네이션 쩔지만 그나마 성능이 나옴)
kogpt2_tokenizer = PreTrainedTokenizerFast.from_pretrained(
    kogpt2_model_name,
    bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)
kogpt2_model = GPT2LMHeadModel.from_pretrained(kogpt2_model_name).to("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
# 2. OpenAI API 키 설정
openai.api_key = ""


# 3. 텍스트 전처리 함수
def clean_text(text):
    # 특수문자 및 불필요한 기호 제거
    text = re.sub(r'\n+', ' ', text)  # 줄바꿈 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9.,?!\s]', '', text)  # 한글, 영어, 숫자, 기본 기호만 남기기
    return text.strip()


# 긴 텍스트 입력 (뉴스 기사나 논문)
text = """
2차 세계대전은 1939년부터 1945년까지 진행된 인류 역사에서 가장 광범위하고 치명적인 전쟁 중 하나로, 전 세계적으로 수억 명의 생명을 앗아갔고, 정치, 경제, 사회, 문화 등 다양한 측면에서 깊은 영향을 미쳤습니다. 이 전쟁은 단순히 군사적 충돌에 그치지 않고, 인류의 윤리와 도덕, 국제 관계의 패러다임을 바꿔 놓은 사건이었습니다.
전쟁의 배경
2차 세계대전의 배경에는 1차 세계대전 이후의 불안정한 국제 정세가 있습니다. 베르사유 조약은 독일에 가혹한 조건을 부과하였고, 이는 독일 내에서 극심한 불만과 경제적 어려움을 초래했습니다. 이러한 상황은 아돌프 히틀러와 나치당의 부상을 가능하게 했고, 그들은 독일의 재무장과 제국주의적 야망을 내세워 전쟁의 불씨를 지폈습니다. 또한, 일본과 이탈리아와 같은 다른 국가들도 제국주의적 확장을 추구하며 전쟁에 가담하게 되었습니다.
전쟁의 전개
2차 세계대전은 유럽, 아프리카, 아시아, 태평양 등 여러 전선에서 동시에 벌어졌습니다. 유럽에서는 독일의 침공으로 시작된 전투가 주요 사건으로, 폴란드 침공, 프랑스 점령, 소련과의 전쟁 등으로 이어졌습니다. 태평양 전선에서는 일본이 중국과 동남아시아를 침공하며 전쟁이 확산되었습니다. 1941년의 진주만 공격은 미국의 참전을 이끌어내는 결정적인 계기가 되었고, 이는 전쟁의 양상을 크게 변화시켰습니다.
"""

cleaned_text = clean_text(text)


# 4. KoGPT-2를 사용한 요약 생성 (프롬프트 기반)
def summarize_with_kogpt2(text, max_new_tokens=300):
    input_ids = kogpt2_tokenizer.encode(text, return_tensors="pt").to(kogpt2_model.device)
    output = kogpt2_model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,  # 새롭게 생성할 토큰의 최대 수
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # 반복 방지
        top_k=50,
        top_p=0.95,
        temperature=0.7,  # 텍스트 다양성 조절
        do_sample=True  # 샘플링을 사용해 텍스트 생성
    )
    summary = kogpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return summary


# 5. GPT-3을 사용한 요약 생성
def summarize_with_gpt3(text, max_tokens=150):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 긴문장 요약을 도와주는 작가야!"},
            {"role": "user", "content": f"다음 텍스트를 요약해 줘:\n\n{text}"}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


# 6. 요약 생성 및 비교
kogpt2_summary = summarize_with_kogpt2("다음의 문장을 한문장으로 요약해 줘" + cleaned_text)
gpt3_summary = summarize_with_gpt3(cleaned_text)

# 결과 출력
print("KoGPT-2 요약 결과:")
print(kogpt2_summary)
print("\nGPT-3 요약 결과:")
print(gpt3_summary)
