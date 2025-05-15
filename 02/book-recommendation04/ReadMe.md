# Book Letter Agent
### app 실행
최종 버전 (*book-recommendation04) 에서 streamlit run app.py

### agent 상태 및 실행 로그 확인
https://smith.langchain.com 접속

### test 수행
* `set_test_case_list_and_gen_jsonl.py` : test_cases 정의, jsonl dataSet 으로 생성해주는 역할 
  * `book_letter_eval_dataset.jsonl` : `set_test_case_list_and_gen_jsonl` 에 생성된 결과
* 


### fine-tuning 수행
```shell
openai tools fine_tunes.prepare_data -f book_letter_fine_tuning_dataset.jsonl
```

위 명령어를 날리면 fine-tuning 시작된다.

아래 로그 확인가능
```log
openai api fine_tuning.jobs.create -t book_letter_fine_tuning_dataset_prepared.jsonl -m gpt-3.5-turbo-1106
```

튜닝이 완료 후, model id 나옴
```log
openai api fine_tuning.jobs.list
```

튜닝된 모델 호출하기
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