import streamlit as st

# 앱 전체 페이지 타이틀, 레이아웃 설정
st.set_page_config(page_title="EV 대시보드", layout="wide")

# 페이지 리스트 정의
pages = [
   #  st.Page("page/날짜,날씨 정보 추가 버전.py", title="대시보드", icon="📊", default=True),
    st.Page("ev 대시보드구현 4.py", title="합친 대시보드", icon="🤖"),
    st.Page("ML예측 페이지.py", title="머신러닝 예측", icon="🤖"),
]

# 페이지 네비게이션 생성
pg = st.navigation(pages)

# 사용자가 선택한 페이지 실행
pg.run()

