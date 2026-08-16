"""青少年个性化学习 Agent 的课程首页。"""

from uuid import uuid4

import streamlit as st

from src.facade import get_app_status
from src.progress import ProgressDataError, default_progress, load_progress


st.set_page_config(
    page_title="学习 Agent",
    page_icon=":material/explore:",
    layout="wide",
)


def initialize_session_state() -> None:
    """集中初始化首页当前需要的 Session State。"""

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"student-{uuid4().hex}"
    if "progress" not in st.session_state:
        try:
            st.session_state.progress = load_progress()
            st.session_state.progress_error = None
        except ProgressDataError as exc:
            st.session_state.progress = default_progress()
            st.session_state.progress_error = str(exc)


def render_runtime_status() -> None:
    """展示配置状态和可以直接执行的修复步骤。"""

    status = get_app_status()
    if status["ready"]:
        provider = status["model_provider"] or "未识别"
        st.success(f"运行环境已就绪，当前模型供应商：{provider}。")
        return

    st.error("运行环境还差一步，请先完成下面的修复。")
    for error in status["errors"]:
        st.write(f"- {error}")
    if status["missing_files"]:
        st.write("缺少的课程文件：")
        for path in status["missing_files"]:
            st.code(path, language=None)

    with st.expander("家长修复步骤", expanded=True):
        st.markdown(
            "1. 确认当前 Python 版本与 `.python-version` 一致。\n"
            "2. 将 `.env.example` 复制为 `.env`，"
            "填写所选模型供应商的 API Key。\n"
            "3. 如果课程文件缺失，请从原始课程包恢复对应文件。\n"
            "4. 保存后重新启动 Streamlit。"
        )


def render_course_card(
    title: str,
    subtitle: str,
    modules: tuple[str, ...],
    completed_modules: set[str],
) -> None:
    """用原生 Streamlit 组件展示一节课的完成进度。"""

    completed_count = sum(module in completed_modules for module in modules)
    progress_value = completed_count / len(modules)
    with st.container(border=True):
        st.subheader(title)
        st.caption(subtitle)
        st.progress(
            progress_value,
            text=f"已完成 {completed_count}/{len(modules)} 个模块",
        )
        st.write(" · ".join(module.upper() for module in modules))


initialize_session_state()
progress = st.session_state.progress
completed_modules = set(progress["completed_modules"])

st.title("你的学习 Agent")
st.write("从会聊天的 AI 开始，一步步训练出真正懂你的学习助手。")

if st.session_state.progress_error:
    st.warning(
        "课程进度文件暂时无法读取，首页已使用默认进度。"
        f"原因：{st.session_state.progress_error}"
    )

render_runtime_status()

lesson_names = {"lesson_1": "第一课", "lesson_2": "第二课"}
metric_columns = st.columns(3)
metric_columns[0].metric("当前课程", lesson_names[progress["current_lesson"]])
metric_columns[1].metric("已完成模块", f"{len(completed_modules)}/5")
metric_columns[2].metric("最近位置", progress["last_location"].upper())

st.subheader("课程路线")
lesson_columns = st.columns(2)
with lesson_columns[0]:
    render_course_card(
        "第一课：训练你的学习助手",
        "比较三个关卡，理解教学说明书和技能怎样改变助手。",
        ("v0", "v1", "v2"),
        completed_modules,
    )
with lesson_columns[1]:
    render_course_card(
        "第二课：让助手学会查证和复盘",
        "加入知识库和 Workflow，完成有证据的长期学习对话。",
        ("v3", "v4"),
        completed_modules,
    )

st.info("第一课已经可以体验，第二课页面将在下一步接入。")
