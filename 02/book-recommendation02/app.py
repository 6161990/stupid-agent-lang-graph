# app.py

import streamlit as st
import asyncio
from node import create_book_letter_graph  # node.py에 있는 create_book_letter_graph 함수

st.set_page_config(page_title="📚 Book Letter Generator", page_icon="📖")
st.title("📚 북 레터 생성기")

keyword = st.text_input("🔍 추천 키워드를 입력하세요:")

if keyword.strip() == "":
    st.warning("키워드를 입력해주세요!")
    st.stop()


async def run_graph(inputs):
    graph = create_book_letter_graph()

    # 진행 상황 표시를 위한 상태 컨테이너
    status_container = st.container()

    with status_container:
        col1, col2 = st.columns([2, 1])
        with col1:
            status_text = st.empty()
        with col2:
            progress_bar = st.progress(0)

        with st.expander("🔎 상세 진행 상황", expanded=True):
            search_status = st.empty()
            theme_status = st.empty()
            subtheme_status = st.empty()
            write_status = st.empty()
            aggregate_status = st.empty()
            edit_status = st.empty()

    step = 0
    total_steps = 10  # 대략적 총 단계 수 설정

    try:
        async for output in graph.astream(inputs, stream_mode="updates"):
            for key, value in output.items():
                step += 1
                progress_bar.progress(min(step / total_steps, 1.0))
                status_text.text(f"현재 단계: {key}")

                # 세부 상태 업데이트
                if key == "search_keyword":
                    search_status.success("✓ 책 제목 검색 완료")
                elif key == "generated_theme":
                    theme_status.success("✓ 테마 생성 완료")
                elif key == "search_sub_themes":
                    subtheme_status.success("✓ 서브테마 리서치 완료")
                elif key.startswith("write_section"):
                    write_status.success(f"✓ 북레터 섹션 작성 완료 ({key})")
                elif key == "aggregate":
                    aggregate_status.success("✓ 북레터 초안 결합 완료")
                    with st.expander("✉️ 북레터 초안 보기", expanded=False):
                        st.markdown(value['messages'][0].content)
                elif key == "editor":
                    edit_status.success("✓ 최종 북레터 편집 완료")
                    st.markdown("## ✨ 최종 북레터")
                    st.markdown(value['messages'][0].content)

        status_text.success("✅ 북 레터 생성이 완료되었습니다!")

    except Exception as e:
        status_text.error("북 레터 생성 중 오류가 발생했습니다.")
        with st.expander("에러 상세 보기"):
            st.error(f"에러 내용: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# 버튼 눌렀을 때 실행
if st.button("📬 북 레터 생성하기"):
    inputs = {"keyword": keyword}
    asyncio.run(run_graph(inputs))
