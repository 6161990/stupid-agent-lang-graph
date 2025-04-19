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
from tavily import AsyncTavilyClient

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

# Book Letter 테마 생성용 데이터모델
class BookLetterThemeOutput(BaseModel):
    theme: str = Field(description="책 제목 리스트 기반 메인 테마")
    sub_themes: List[str] = Field(description="5개 정도의 구체적인 서브테마")

# Structured Output 세팅
structured_llm_book_letter = llm.with_structured_output(BookLetterThemeOutput)

# Prompt 설정
system_prompt = """
You are an expert editorial assistant for a themed book letter. Your task is to create a consistent and compelling topic structure based on a given list of book titles.

Instructions:
- Carefully examine the book titles and infer the common thread or theme that connects them.
- Choose **one main theme** that reflects this connection. The theme should be written in the form of a specific, thought-provoking question in Korean.
- Then, write **five sub-themes**, each clearly and meaningfully derived from the main theme and the titles.
- All output must be written in Korean.
"""

theme_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "Book titles: \n\n {book_titles}")]
)

book_letter_generator = theme_prompt | structured_llm_book_letter

# State 정의
def merge_dicts(left: Dict, right: Dict) -> Dict:
    return {**left, **right}

class State(TypedDict):
    keyword: str
    book_titles: List[str]
    book_letter_theme: BookLetterThemeOutput
    sub_theme_books: Dict[str, List[Dict]]
    results: Annotated[Dict[str, str], merge_dicts]
    messages: Annotated[List, add_messages]

# Node 함수들
def search_keyword(state: State) -> State:
    keyword = state['keyword']
    book_titles = search_for_book_titles(keyword)
    return {"book_titles": book_titles}

def generated_book_letter_theme(state: State) -> State:
    book_titles = state['book_titles']
    book_letter_theme = book_letter_generator.invoke({"book_titles": "\n".join(book_titles)})
    book_letter_theme.sub_themes = book_letter_theme.sub_themes[:5]
    return {"book_letter_theme": book_letter_theme}

async def search_sub_themes(state: State) -> State:
    sub_themes = state['book_letter_theme'].sub_themes
    results = await asyncio.gather(*[search_sub_theme(sub_theme) for sub_theme in sub_themes])

    sub_theme_books = {}
    for result in results:
        sub_theme_books.update(result)

    return {"sub_theme_books": sub_theme_books}

async def search_sub_theme(sub_theme):
    async_tavily_client = AsyncTavilyClient()
    response = await async_tavily_client.search(
        query=sub_theme,
        max_results=3,
        topic="general",
        days=100,
        include_images=True,
        include_raw_content=True
    )
    images = response['images']
    results = response['results']

    book_letter_info = []
    for i, result in enumerate(results):
        book_letter_info.append({
            'title': result['title'],
            'image_url': images[i],
            'raw_content': result['raw_content'],
        })
    return {sub_theme: book_letter_info}

def write_book_letter_section(state: State, sub_theme: str) -> Dict:
    return asyncio.run(write_book_letter_section_async(state, sub_theme))

async def write_book_letter_section_async(state: State, sub_theme: str) -> Dict:
    books = state['sub_theme_books'][sub_theme]
    books_references = "\n".join(
        [f"Title: {book['title']}\nContent: {book['raw_content']}..." for book in books]
    )

    prompt = f"""
    Write a book letter section for the sub-theme: "{sub_theme}".
    Use the following books as reference:
    <book>
    {books_references}
    </book>
    Write in Korean, warmly and engaging.
    """

    messages = [HumanMessage(content=prompt)]
    response = await llm.ainvoke(messages)
    return {"results": {sub_theme: response.content}}

def aggregate_results(state: State) -> State:
    theme = state['book_letter_theme'].theme
    combined_book_letter = f"# {theme}\n\n"
    for sub_theme, content in state['results'].items():
        combined_book_letter += f"## {sub_theme}\n{content}\n"
    return {"messages": [HumanMessage(content=f"Generated Book Letter:\n\n{combined_book_letter}")]}

def edit_book_letter(state: State) -> State:
    theme = state['book_letter_theme'].theme
    combined_book_letter = state['messages'][-1].content

    prompt = f"""
You are writing a book letter that should feel like a heartfelt personal message.

Theme: "{theme}"

{combined_book_letter}

Make it warm, emotional, personal, handwritten-style.
Output in Korean only.
"""

    writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=1, max_tokens=8192)
    messages = [HumanMessage(content=prompt)]
    response = writer_llm.invoke(input=messages)

    return {"messages": [HumanMessage(content=f"Edited Book Letter:\n\n{response.content}")]}

# Graph 구성 함수
def create_book_letter_graph():
    workflow = StateGraph(State)
    workflow.add_node("search_keyword", search_keyword)
    workflow.add_node("generated_theme", generated_book_letter_theme)
    workflow.add_node("search_sub_themes", search_sub_themes)
    for i in range(5):
        workflow.add_node(f"write_section_{i}", lambda s, i=i: write_book_letter_section(s, s['book_letter_theme'].sub_themes[i]))
    workflow.add_node("aggregate", aggregate_results)
    workflow.add_node("editor", edit_book_letter)

    workflow.add_edge(START, "search_keyword")
    workflow.add_edge("search_keyword", "generated_theme")
    workflow.add_edge("generated_theme", "search_sub_themes")
    for i in range(5):
        workflow.add_edge("search_sub_themes", f"write_section_{i}")
        workflow.add_edge(f"write_section_{i}", "aggregate")
    workflow.add_edge("aggregate", "editor")
    workflow.add_edge("editor", END)

    return workflow.compile()
