import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langsmith.client import Client
from node import create_book_letter_graph  # 네가 만든 LangGraph 함수
from evaluator import smart_evaluate

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
    evaluators=[smart_evaluate],
    experiment_prefix="book-letter-eval",
    description="주어진 카테고리 내에서만 정의하도록 프롬포트 명확하게 가이드"
)

print("✅ LangStudio 자동 평가 실행 완료!")


# 5. 평가 결과 기반 fine-tuning 데이터셋 추출
fine_tune_data = []

runs = client.list_runs(
    project_name="evaluators",
    execution_order=1,
    run_type="chain"
)

for run in runs:
    feedbacks = client.list_feedback(run_id=run.id)
    scores = {fb.key: fb.score for fb in feedbacks}

    if scores.get("category_match") == 1.0 and scores.get("embedding_distance_score", 0) >= 0.9:
        example_id = run.extra.get("metadata").get("reference_example_id")
        example = client.read_example(example_id)

        input_text = example.inputs.get("keyword")
        output_text = example.outputs.get("messages")[0].get("content")

        fine_tune_data.append({
            "messages": [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ],
            "metadata": {
                "run_id": str(run.id),
                "scores": {k: float(v) if isinstance(v, (int, float)) else bool(v) for k, v in scores.items() if isinstance(v, (int, float, bool, type(None)))}
            }
        })

# 6. 저장
output_path = "book_letter_fine_tuning_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in fine_tune_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Fine-tuning 데이터셋 저장 완료: {output_path}")