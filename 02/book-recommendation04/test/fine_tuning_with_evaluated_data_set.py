import json
import os
from dotenv import load_dotenv
from langsmith.client import Client

# 환경변수 로드
load_dotenv()

langchain_traceing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_key = os.getenv("LANGCHAIN_API_KEY")

client = Client()

# 1. 평가 결과 기반 fine-tuning 데이터셋 추출
fine_tune_data = []

runs = client.list_runs(
    project_name="evaluators",
    execution_order=1,
    run_type="chain"
)

for run in runs:
    feedbacks = client.list_feedback(run_id=run.id)
    scores = {fb.key: fb.score for fb in feedbacks}

    if scores.get("category_match") == 1.0 and scores.get("embedding_distance_score", 0) >= 0.8:
        metadata = run.extra.get("metadata", {})
        example_id = metadata.get("reference_example_id")

        if not example_id:
            continue  # reference가 없는 경우 skip

        try:
            example = client.read_example(example_id)
        except Exception as e:
            print(f"⚠️ 예시 불러오기 실패: {example_id} - {e}")
            continue
        input_text = example.inputs.get("keyword")
        output_messages = example.outputs.get("messages", [])

        if input_text and output_messages and hasattr(output_messages[0], "content"):
            output_text = output_messages[0]["content"]

            fine_tune_data.append({
                "prompt": input_text.strip(),
                "completion": output_text.strip() + "\n"  # OpenAI 규칙: \n 마무리
            })

# 2. 저장
output_path = "book_letter_fine_tuning_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for item in fine_tune_data:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Fine-tuning 데이터셋 저장 완료: {output_path}")
