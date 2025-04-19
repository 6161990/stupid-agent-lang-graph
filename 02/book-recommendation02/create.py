from langgraph.graph import StateGraph, END, START
from node import State
from node import (
    edit_book_letter,
    generated_book_letter_theme,
    search_keyword,
    search_sub_themes,
    aggregate_results,
    write_book_letter_section,
)
import logging

logger = logging.getLogger(__name__)


def create_book_letter_graph():
    logger.info("Creating book letter graph")

    workflow = StateGraph(State)

    # 노드 추가
    workflow.add_node("editor", edit_book_letter)
    workflow.add_node("search_keyword", search_keyword)
    workflow.add_node("generate_theme", generated_book_letter_theme)
    workflow.add_node("search_sub_themes", search_sub_themes)
    workflow.add_node("aggregate", aggregate_results)

    for i in range(5):
        node_name = f"write_section_{i}"
        workflow.add_node(
            node_name,
            lambda s, i=i: write_book_letter_section(s, s['book_letter_theme'].sub_themes[i])
        )

    # 엣지 연결
    workflow.add_edge(START, "search_keyword")  # 오타 수정
    workflow.add_edge("search_keyword", "generate_theme")
    workflow.add_edge("generate_theme", "search_sub_themes")

    for i in range(5):
        workflow.add_edge("search_sub_themes", f"write_section_{i}")
        workflow.add_edge(f"write_section_{i}", "aggregate")

    workflow.add_edge("aggregate", "editor")
    workflow.add_edge("editor", END)

    logger.info("Book letter graph created successfully")
    return workflow.compile()
