"""青少年个性化学习 Agent 的 Streamlit 入口。"""

import streamlit as st


page = st.navigation(
    [
        st.Page(
            "app_pages/home.py",
            title="课程首页",
            icon=":material/home:",
            default=True,
        ),
        st.Page(
            "pages/1_prompt_and_skill.py",
            title="第一课：训练学习助手",
            icon=":material/school:",
        ),
    ],
    position="sidebar",
)
page.run()
