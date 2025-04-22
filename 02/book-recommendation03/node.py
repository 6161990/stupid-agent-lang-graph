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
    4. 편지 제목은 '질문 형태'로 작성합니다. (예: "당신에게 작은 위로가 되어줄까요?")
    5. 편지의 마지막은 별도의 수신인 언급 없이 자연스럽게 마무리합니다.

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
    "지친 하루 끝에 작은 위로가 필요할까요?"

    <본문>  
    오늘 하루도 고생 많으셨습니다. 때로는 말보다 책 한 권이 마음을 토닥여줄 때가 있습니다.  
    『연년세세』에서는 사소한 일상 속에서도 소중함을 발견하는 법을 다정히 이야기해줍니다. 삶에 지친 당신이라면 이 책에서 조용한 위안을 얻을 수 있을 거예요.  
    또한 『모든 순간의 물리학』은 과학이라는 렌즈를 통해 세상을 다정하게 들여다봅니다. 물리학이 이렇게 포근할 수 있다는 사실에 놀랄지도 모릅니다.  
    혹시 더 깊은 생각에 잠기고 싶다면, 『죽음에 관하여』를 펼쳐보세요. 인생의 끝을 따뜻하게 바라보는 시선을 통해 오늘을 더욱 소중히 느낄 수 있습니다.  

    오늘은 이렇게 작은 쉼표 하나, 책과 함께 선물해보는 건 어떨까요?

    ---

    [예시: 전문적 톤 편지 샘플]

    <제목>  
    "빠르게 변화하는 시대, 당신은 어떤 선택을 할 준비가 되어 있나요?"

    <본문>  
    지금 우리는 기술과 경제가 눈부시게 변화하는 시대에 살고 있습니다. 『플랫폼 제국의 미래』는 이 거대한 변화의 흐름 속에서 플랫폼 기업들이 어떻게 세상을 재편하는지 깊이 있게 분석합니다. 실리콘밸리의 전략과 의도를 읽고 싶은 분들에게 필독서입니다.  
    『예측가능한 불확실성』은 오늘날 비즈니스 리더들이 직면한 불확실성을 어떻게 기회로 전환할 수 있는지를 알려줍니다. 특히 스타트업이나 변화 관리에 관심 있는 분들에게 큰 도움이 될 것입니다.  
    마지막으로 『테크노 사피엔스』는 기술 혁신이 인간 사회에 끼치는 장기적 영향을 성찰하게 해줍니다. 단순한 기술 소개를 넘어, 기술과 인간의 관계를 깊이 고민해보고 싶은 독자에게 권합니다.  

    급변하는 세상 속에서 통찰력을 키우고 싶다면, 이 책들과 함께 준비해보세요.

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
