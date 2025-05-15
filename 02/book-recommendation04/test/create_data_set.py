import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langsmith.client import Client
from node import create_book_letter_graph  # 네가 만든 LangGraph 함수
from evaluator import evaluate_all

# 환경변수 로드
load_dotenv()

langchain_traceing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_key = os.getenv("LANGCHAIN_API_KEY")

client = Client()
strftime = datetime.now().strftime("%Y%m%d-%H%M%S")
dataset_name = "book-letter-eval" + strftime
dataset = client.create_dataset(
    dataset_name=("%s" % dataset_name),
    description="북레터 에이전트 프롬프트 평가용 테스트셋" + strftime
)

# JSONL 파일을 한 줄씩 읽어서 업로드
with open("book_letter_eval_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        client.create_example(
            dataset_id=dataset.id,
            inputs=item["inputs"],
            outputs=item.get("expected_output")
        )

print("✅ LangStudio Dataset 업로드 완료")


# 에이전트 실행용 함수 정의
def run_agent(inputs: dict) -> dict:
    graph = create_book_letter_graph()
    return graph.invoke(inputs)


client.evaluate(
    run_agent,
    data=dataset_name,
    evaluators=[evaluate_all],
    experiment_prefix="book-letter-eval",
    description="주어진 카테고리 내에서만 정의하도록 프롬포트 명확하게 가이드"
)

print("✅ LangStudio 자동 평가 실행 완료!")

