## ReadMe

## 실행 전제 조건
.env 에 아래 key 에 대한 value 세팅 필요
- OPENAI_API_KEY
- NAVER_CLIENT_ID
- NAVER_CLIENT_SECRET
- langchain_traceing_v2
- langchain_endpoint
- langchain_key

*[chatGPT 모델 사용 권한 키 관리 페이지](https://platform.openai.com/api-keys) 에서 신규 키 발급받을 수 있음.
현재 잔금 남아있음   
*[파인튜닝도 가능](https://platform.openai.com/finetune)


### app.py
streamlit 으로 연동된 북레터 에이전트 애플리케이션 main

### node.py
langGraph 로 구현한 북레터 에이전트 노드

### book_letter_prompt.py
- 북레터 에이전트 프롬포트

### book_letter_rules.py
- 북레터 규칙 가이드. 
- 키워드 관련 북레터 톤(tone) 정의

### define_test_cases_and_generate_file.py
- 테스트 케이스 정의 
- messages 필드를 추가한 자동 평가용 + 커스텀 평가용 복합 테스트 셋
- book_letter_eval_dataset.jsonl 생성

### book_letter_eval_dataset.jsonl
- define_test_cases_and_generate_file.py 에 의해 생성된 테스트 셋 파일
- langSmith 가 해당 파일 내용을 기반으로 평가 UI 생성

### create_data_set.py
- book_letter_eval_dataset.jsonl 을 읽고 langSmith 가 dataSet 을 생성
- evaluator.py 을 넘겨서 해당 테스트 셋에 대한 평가를 진행할 수 있음

### evaluator.py
- 테스트 셋에 대한 평가 기준 정의
- Exact Match / Embedding Similarity 자동 평가 방식
- Rubric 기반의 Human Eval 수동 평가 방식 


### streamlit 실행 방법
```shell
# 실행하려는 애플리케이션이 위치한 루트에서  
streamlit run app.py
```


### 테스트 셋에 대한 평가를 실행하고 싶다면?
1. define_test_cases_and_generate_file.py 에 테스트 케이스 정의 후 실행
2. 테스트 셋 파일 생성되었다면, create_date_set.py 에서 테스트 평가 실행 
3. [langSmith](https://smith.langchain.com/studio/thread?baseUrl=http%3A%2F%2F127.0.0.1%3A2024&organizationId=93533745-b108-4cf2-88f9-430dc097993e#) 접속하여 테스트 평가 확인 


### 테스트 결과를 통해 우수한 케이스 선별하여 fine-tuning 하고 싶다면?
1. fine_tuning_with_evaluated_data_set.py 실행
2. book_letter_fine_tuning_dataset.jsonl 파일 생성되었다면, 아래 command 실행
```shell
openai tools fine_tunes.prepare_data -f book_letter_fine_tuning_dataset.jsonl
```
3. 아래 log 확인
![img.png](img/img.png)
4. 아래 command 실행하여 파인튜닝 수행. 
```shell
openai api fine_tuning.jobs.create -t book_letter_fine_tuning_dataset_prepared.jsonl -m gpt-3.5-turbo-1106
```

5. 튜닝된 모델 호출하기
```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-1106:your-org::custom-id",
    messages=[
        {"role": "user", "content": "이별 후 위로"}
    ]
)

print(response.choices[0].message.content)
```