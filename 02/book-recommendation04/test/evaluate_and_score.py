import json

from node import create_book_letter_graph
from evaluator import evaluate_letter

graph = create_book_letter_graph()

# 테스트셋 불러오기와 수동 평가
with open("book_letter_eval_dataset.jsonl", encoding="utf-8") as f:
    test_cases = [json.loads(line) for line in f]

for idx, case in enumerate(test_cases, 1):
    keyword = case["inputs"]["keyword"]
    criteria = case["expected_output"]["criteria"]
    expected_category = case["expected_output"]["category"]

    inputs = {"keyword": keyword}
    state = graph.invoke(inputs)

    actual_letter = state["messages"][0].content
    actual_category = state["category"]

    print(f"\n🧪 [{idx}] '{keyword}' 테스트:")
    print(f"   - 예상 카테고리: {expected_category}")
    print(f"   - 판별된 카테고리: {actual_category} ✅" if actual_category == expected_category else f"{actual_category} ❌")

    eval_result = evaluate_letter(actual_letter, criteria)
    for k, v in eval_result.items():
        print(f"   - {k}: {'✅' if v else '❌'}")

