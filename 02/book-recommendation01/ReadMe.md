### ReadMe

#### class State(TypedDict)
상태 그래프에서 사용하는 전역 상태를 정의한 딕셔너리 타입.

* keyword: 사용자가 입력한 키워드
* book_titles: 키워드로부터 관련된 책 제목 리스트
* book_letter_theme: 북레터의 메인 주제와 소주제들이 포함된 객체
* sub_theme_books: 소주제별로 검색된 책 정보 딕셔너리
* results: 기타 결과
* messages: LangGraph 내부 메시지 히스토리


#### 🔹 generated_book_letter_theme(state: State) -> State
book_titles 리스트를 기반으로 북레터 테마 및 소주제를 생성.
book_letter_generator 라는 외부 모델 호출을 통해 테마 추출.
예: "인간관계"라는 테마에서 "공감", "갈등", "대화법" 등의 소주제 뽑기



#### 1.0.0 버전에서 디벨롭 하고 싶은 것 

- 누군가가 누군가에게 전달하는 편지 형식의 북레터 느낌이 좀 더 진했으면 좋겠음. 
- 열 줄 이하로 적었으면 좋겠음. 
- 비슷한 작품의 문학/ 인문학/ 사회 / 에세이 로 한정하여 추천이 있었으면 좋겠음 

