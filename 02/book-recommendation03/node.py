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

    # 프롬프트 설계
    prompt = f"""
너는 친근하고 따뜻한 책 추천 편지를 작성하는 에디터야.

주제 키워드: "{keyword}"

아래 책 제목들을 자연스럽게 소개하고 추천해줘.
책마다 간단한 추천 이유도 살짝 곁들여서.
북레터 제목은 부드러운 질문 형태로 작성할 것.

책 목록:
{chr(10).join(f"- {title}" for title in book_titles)}

출력은 자연스러운 한국어로 작성할 것.
"""

    writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    messages = [HumanMessage(content=prompt)]
    response = writer_llm.invoke(messages)

    return {"messages": [HumanMessage(content=f"Generated Book Letter:\n\n{response.content}")]}


# Graph 구성
def create_book_letter_graph():
    workflow = StateGraph(State)

    workflow.add_node("search_book_titles", search_book_titles)
    workflow.add_node("write_book_letter", write_book_letter)

    workflow.add_edge(START, "search_book_titles")
    workflow.add_edge("search_book_titles", "write_book_letter")
    workflow.add_edge("write_book_letter", END)

    return workflow.compile()
