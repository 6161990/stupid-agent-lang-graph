from typing import List

category1 = ["문학", "예술", "철학", "감성"]
category2 = ["경제", "산업", "사회", "기술", "정치"]

def get_usable_category() -> List[str]:
    return category1 + category2

def get_tone_style(category: str) -> str:
    if category in category1:
        return "부드럽고 감성적인 톤"
    elif category in category2:
        return "신뢰감 있고 전문적인 톤"
    else:
        return "기본적으로 따뜻한 감성 톤"
