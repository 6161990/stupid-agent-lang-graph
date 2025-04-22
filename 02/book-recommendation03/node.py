import os
import requests
import asyncio
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# 환경변수 로드
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
langchain_traceing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_key = os.getenv("LANGCHAIN_API_KEY")


# 책 검색 함수
def search_for_book_titles_naver(keyword):
    url = "https://openapi.naver.com/v1/search/book.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": keyword,
        "display": 5,
        "sort": "sim"
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    items = response.json().get("items", [])
    return [item["title"].replace("<b>", "").replace("</b>", "") for item in items]


def search_for_book_titles_google(keyword: str, max_results: int = 5) -> List[str]:
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": keyword,
        "maxResults": max_results,
        "printType": "books",
        "langRestrict": "ko",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    items = response.json().get("items", [])
    return [item["volumeInfo"].get("title", "제목 없음") for item in items]


def search_for_book_titles(keyword):
    titles = search_for_book_titles_google(keyword)
    if not titles:
        titles = search_for_book_titles_naver(keyword)
    return titles


# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)


# State 정의
def merge_dicts(left: Dict, right: Dict) -> Dict:
    return {**left, **right}


class State(TypedDict):
    keyword: str
    book_titles: List[str]
    messages: Annotated[List, add_messages]


# Node 함수들 (축소된 버전)

def search_book_titles(state: State) -> State:
    keyword = state['keyword']
    book_titles = search_for_book_titles(keyword)
    return {"book_titles": book_titles}


def write_book_letter(state: State) -> State:
    keyword = state['keyword']
    book_titles = state['book_titles']

    cleaned_titles = [title.strip("'\"") for title in book_titles]
    book_list = "\n".join(f"- <{title}>" for title in cleaned_titles)

    # 프롬프트 설계
    prompt = f"""
    당신은 '상황에 맞는 톤과 스타일을 조정하여 독자에게 책을 추천하는 전문 에디터'입니다.

    [당신의 역할]
    - 주어진 키워드와 책 목록을 기반으로, 독자에게 전달할 부드러운 '책 추천 편지'를 작성합니다.

    [작성 작업]
    1. 편지 서두에서는 자연스럽게 독자에게 말을 거는 따뜻한 인삿말로 시작합니다.
    2. 책 제목을 단순 나열하지 않고, 하나의 이야기처럼 부드럽게 연결하여 소개합니다.
    3. 각 책에 대해:
        - 짧은 추천 이유를 덧붙입니다.
        - 적절한 경우, 책 내용이나 등장인물을 간단히 인용할 수 있습니다.
        - 특정 상황(예: 치유가 필요한 사람, 새로운 아이디어를 찾는 사람 등)이나 독자에게 연결합니다.
    4. 누구나 할 수 있는 "좋은 책입니다" 수준에 머무르면 안됩니다. 너무 일반적 표현은 지양하세요.
    5. 구체적인 사례나 장면을 추가하여 깊이있게 소개하도록 하세요.
    6. 읽는 사람 입장에서 "내 삶에 적용하면 어떨까" 를 생각할 수 있게끔 물리적이고 현실적인 언어를 추가해주세요. 
    7. 편지의 마지막은 별도의 수신인 언급 없이 자연스럽게 마무리합니다.


    [톤과 스타일 규정]
    - 키워드가 ["문학", "예술", "철학", "감성"] 관련이면:
        - 부드럽고 감성적인 톤을 사용하세요.
    - 키워드가 ["경제", "산업", "사회", "기술", "정치"] 관련이면:
        - 신뢰감 있고 전문적인 톤을 사용하되, 친절하고 부드러운 느낌은 유지하세요.
    - 키워드를 명확히 분류할 수 없는 경우, 기본적으로 따뜻한 감성 톤을 유지하세요.

    [추가 지침]
    - 한국어로 자연스럽고 부드러운 문장 스타일을 사용하세요.
    - 가능하다면, 교보문고 사이트에서 해당 책의 구매 링크를 자연스럽게 연결하세요.
        (단, 문맥을 깨지 않도록 주의하세요. 억지로 삽입하지 마세요.)

    [예시: 따뜻한 톤 편지 샘플]

    <제목>  
    "지금 나와 같은 누군가를 찾고 있나요?"

    <본문>  
    어디선가 나와 같은 누군가가 살고 있다면, 그건 우리에게 큰 위로가 될지도 모릅니다.   
    서로 다른 곳에 있어도, 비슷한 생각을 품고 있는 사람들이 있다는 것만으로 우리는 조금 더 단단해질 수 있지 않을까요.
    그런 마음을 닮은 책이 있습니다. 『연년세세』는 사소한 일상 속에서도 소중함을 발견하는 법을 다정히 이야기합니다. 반복되는 하루에 지쳐 있을 때, 이 책은 작은 숨구멍이 되어줄 거예요.
    과학이라는 렌즈로 세상을 부드럽게 바라보고 싶다면 『모든 순간의 물리학』을 추천하고 싶습니다. 복잡하고 어려운 이야기가 아니라, 우리가 매일 스쳐 지나가는 순간들 속 물리학의 따뜻한 얼굴을 발견할 수 있습니다.
    그리고 언젠가 찾아올 이별과 끝에 대해 조심스럽게 생각하고 싶다면, 『죽음에 관하여』를 펼쳐보세요. 죽음이라는 주제를 담담하고 깊이 있게 풀어내며, 오히려 지금 이 순간을 소중히 살아가게 만들어주는 책입니다.
    오늘 당신의 하루에, 이 책들이 작은 위로가 되어줄 수 있기를 바랍니다.

    ---

    [예시: 전문적 톤 편지 샘플]

    <제목>  
    "당신은 어떤 선택을 할 준비가 되어 있나요?"

    <본문>  
    어제 만든 전략이 오늘은 낡은 이야기가 됩니다. 아침에 결정한 방향이 저녁이면 수정되어야 하는 시대.
    『플랫폼 제국의 미래』는 이런 변화의 정중앙에 서 있는 플랫폼 기업들의 전략을 해부합니다. 아마존이 물류를 장악하고, 애플이 생태계를 설계하며, 구글이 검색을 넘어 인공지능 시장을 점령하는 과정을 사례 중심으로 보여줍니다. 단순한 기업 소개가 아니라, 왜 이들이 "제국"이 되었는지 궤적을 따라갑니다.
    『예측가능한 불확실성』은 스타트업 창업자들이 초기에 겪는 혼란, 투자자와의 충돌, 팀 빌딩 실패 같은 구체적 상황을 풀어내며, 불확실한 상황에서도 어떻게 방향을 잃지 않는지 실질적인 조언을 건넵니다. 회의실에서 끝없는 의견 충돌을 겪고 있는 당신이라면, 이 책이 필요한 나침반이 되어줄 겁니다.
    마지막으로 『테크노 사피엔스』는 자율주행차, 유전자 편집, 초지능 AI 같은 기술들이 인간 사회를 어떻게 뒤흔들지 생생하게 묘사합니다. 기술 혁신을 단순한 희망이나 위협이 아니라, 인간 존재에 대한 깊은 질문으로 끌어올립니다. 실리콘밸리 뉴스 헤드라인 너머의 진짜 변화를 읽고 싶은 독자라면 꼭 손에 쥐어야 할 책입니다.
    급변하는 세상 속에서, 진짜 중요한 질문을 던지고 싶다면, 이 책들과 함께 준비해보세요.

    ---

    [제공된 책 목록]
    {book_list}

    지금부터 위의 지침과 예시를 참고하여, 주어진 키워드와 책 목록을 기반으로 책 추천 편지를 작성해주세요.
    """

    writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    messages = [HumanMessage(content=prompt)]
    response = writer_llm.invoke(messages)

    return {"messages": [HumanMessage(content=f"\n\n{response.content}")]}


# Graph 구성
def create_book_letter_graph():
    workflow = StateGraph(State)

    workflow.add_node("search_book_titles", search_book_titles)
    workflow.add_node("write_book_letter", write_book_letter)

    workflow.add_edge(START, "search_book_titles")
    workflow.add_edge("search_book_titles", "write_book_letter")
    workflow.add_edge("write_book_letter", END)

    return workflow.compile()
