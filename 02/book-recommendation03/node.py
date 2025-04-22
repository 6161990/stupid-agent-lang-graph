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
    너는 상황에 맞춰 글의 톤과 스타일을 조정하는 섬세한 책 추천 편지 에디터야.

    [전체 작성 규칙]
    - 하나의 부드러운 편지 형식으로 책들을 자연스럽게 소개해야 해.
    - 책 제목을 단순 나열하지 말고 이야기 흐름처럼 자연스럽게 연결해줘.
    - 각 책은 짧은 추천 이유와 함께 자연스럽게 녹여내고, 특정 상황이나 독자에게 연결해줘.
    - 필요하다면 책 내용 중 인상 깊은 구절이나 등장인물 이름을 인용해도 좋아.
    - 편지 제목은 질문 형태로 작성할 것. (예: "당신에게 작은 위로가 되어줄까요?")
    - 편지 마지막에는 수신인에 대한 언급 없이 마무리할 것.

    [특별 요청 - 키워드 기반 톤 설정]
    - 키워드가 "문학", "예술", "철학", "감성"과 관련되었다면: 지금처럼 따뜻하고 감성적인 톤으로 작성해줘.
    - 키워드가 "경제", "산업", "사회", "기술", "정치"처럼 전문 분야라면: 조금 더 신뢰감 있고 전문적인 어조로 써줘. (단, 너무 딱딱하지 않고 친절한 느낌 유지)
    - 만약 키워드 성격을 판단하기 애매할 경우, 따뜻한 톤을 기본으로 유지해줘.

    [책 목록]
    {book_list}

    [추가 요청]
    - 자연스럽고 부드러운 한국어 스타일을 유지할 것.
    - 교보문고 사이트에서 해당 책의 구매 링크가 존재한다면 자연스럽게 연결해서 소개할 것. (억지스럽게 끼워 넣지 말 것)

    지금부터 편지를 시작해줘.
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
