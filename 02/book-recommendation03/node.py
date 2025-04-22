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

    [제공된 책 목록]  
    {book_list}  

    지금부터 위의 기준에 맞춰 책 추천 편지를 작성해주세요.  
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
