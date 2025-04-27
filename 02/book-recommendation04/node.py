import os
import requests
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from prompt.book_letter_prompt import build_book_letter_prompt
from rules.book_letter_rules import get_usable_category

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
    titles = search_for_book_titles_naver(keyword)
    if not titles:
        titles = search_for_book_titles_google(keyword)
    return titles


# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)


# State 정의
def merge_dicts(left: Dict, right: Dict) -> Dict:
    return {**left, **right}


class State(TypedDict):
    keyword: str
    book_titles: List[str]
    category: str
    messages: Annotated[List, add_messages]



# Node 함수들 (축소된 버전)
def detect_category(state: State) -> State:
    keyword = state["keyword"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = f"""
    다음 키워드를 보고, 가장 적절한 카테고리를 선택하세요.
    가능한 카테고리: {get_usable_category}
    단 하나만 골라야 하며, 답변은 카테고리 이름만 출력하세요.
    """

    prompt = f"{system_prompt}\n\n키워드: {keyword}"

    response = llm.invoke([HumanMessage(content=prompt)])
    category = response.content.strip()

    return {"category": category}

def search_book_titles(state: State) -> State:
    keyword = state['keyword']
    book_titles = search_for_book_titles(keyword)
    return {"book_titles": book_titles}


def write_book_letter(state: State) -> State:
    category = state['category']
    book_titles = state['book_titles']

    cleaned_titles = [title.strip("'\"") for title in book_titles]
    book_list = "\n".join(f"- <{title}>" for title in cleaned_titles)

    prompt = build_book_letter_prompt(category, book_list)

    writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    messages = [HumanMessage(content=prompt)]
    response = writer_llm.invoke(messages)

    return {"messages": [HumanMessage(content=f"\n\n{response.content}")]}



# Graph 구성
def create_book_letter_graph():
    workflow = StateGraph(State)
    workflow.tracing_v2 = True

    workflow.add_node("search_book_titles", search_book_titles)
    workflow.add_node("detect_category", detect_category)  # 추가
    workflow.add_node("write_book_letter", write_book_letter)

    workflow.add_edge(START, "search_book_titles")
    workflow.add_edge("search_book_titles", "detect_category")  # 연결 추가
    workflow.add_edge("detect_category", "write_book_letter")
    workflow.add_edge("write_book_letter", END)

    return workflow.compile()
