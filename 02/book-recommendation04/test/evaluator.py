from langchain_community.utils.math import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from typing import List

def evaluate_letter(letter: str, criteria: dict) -> List[dict]:
    results = []

    if criteria.get("should_start_with_greeting"):
        greetings = ["안녕하세요", "오늘 하루도", "어디선가", "지친 하루"]
        matched = any(letter.strip().startswith(g) for g in greetings)
        results.append({
            "key": "starts_with_greeting",
            "score": matched,
            "comment": f"기준 'starts_with_greeting' 평가 결과: {'통과' if matched else '실패'}"
        })

    if criteria.get("should_end_without_recipient"):
        forbidden = ["감사합니다", "드림", "당신의", "올림"]
        matched = not any(f in letter[-50:] for f in forbidden)
        results.append({
            "key": "ends_without_recipient",
            "score": matched,
            "comment": f"기준 'ends_without_recipient' 평가 결과: {'통과' if matched else '실패'}"
        })

    if criteria.get("should_use_question_title"):
        title_line = letter.strip().split("\n")[0]
        matched = "?" in title_line
        results.append({
            "key": "question_title",
            "score": matched,
            "comment": f"기준 'question_title' 평가 결과: {'통과' if matched else '실패'}"
        })

    if "book_title_count" in criteria:
        count = letter.count("『")
        required = criteria["book_title_count"]
        matched = count >= required
        results.append({
            "key": "book_title_count",
            "score": matched,
            "comment": f"기준 'book_title_count' 평가 결과: {count}/{required}"
        })

    if criteria.get("should_include_concrete_examples"):
        keywords = ["사례", "경험", "현장", "현실", "살면서"]
        matched = any(k in letter for k in keywords)
        results.append({
            "key": "concrete_examples",
            "score": matched,
            "comment": f"기준 'concrete_examples' 평가 결과: {'통과' if matched else '실패'}"
        })

    if "tone" in criteria:
        tone = criteria["tone"]
        if tone == "감성":
            tone_keywords = ["위로", "마음", "조용히", "토닥"]
        elif tone == "전문적":
            tone_keywords = ["전략", "경영", "데이터", "분석"]
        else:
            tone_keywords = []
        matched = any(k in letter for k in tone_keywords)
        results.append({
            "key": "tone_match",
            "score": matched,
            "comment": f"기준 'tone_match' 평가 결과: {'통과' if matched else '실패'}"
        })

    return results


# 유사도 계산
def compute_similarity(text1: str, text2: str) -> float:
    embedder = OpenAIEmbeddings()
    vec1 = embedder.embed_query(text1)
    vec2 = embedder.embed_query(text2)
    return float(cosine_similarity([vec1], [vec2])[0][0])

def evaluate_all(run, example) -> List[dict]:
    results = []

    expected_category = example.outputs.get("category")
    actual_category = run.outputs.get("category")
    if expected_category is not None and actual_category is not None:
        results.append({
            "key": "category_match",
            "score": expected_category == actual_category,
            "comment": "카테고리 정확도 평가"
        })

    try:
        expected = example.outputs.get("messages", [{}])[0].get("content", "")
        actual_msgs = run.outputs.get("messages", [])
        actual = actual_msgs[0].content if actual_msgs and hasattr(actual_msgs[0], "content") else ""

        if expected and actual:
            sim = compute_similarity(actual, expected)
            results.append({
                "key": "embedding_distance_score",
                "score": sim,
                "comment": "북레터 의미 유사도 평가"
            })
    except Exception as e:
        results.append({
            "key": "embedding_distance_score",
            "score": None,
            "comment": f"유사도 평가 실패: {str(e)}"
        })

    if example.outputs.get("criteria"):
        letter_criteria = evaluate_letter(actual, example.outputs["criteria"])
        results.extend(letter_criteria)

    return results