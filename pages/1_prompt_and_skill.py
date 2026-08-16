"""第一课：用聊天体验 Prompt 和 Skill 如何改变学习助手。"""

from collections.abc import Iterator
from contextlib import contextmanager
from difflib import unified_diff

import streamlit as st

from src.facade import (
    LessonArtifactError,
    get_lesson_artifact,
    invoke,
    save_lesson_artifact,
)
from src.progress import (
    ProgressDataError,
    complete_module,
    default_progress,
    load_progress,
    save_progress,
)


STAGES = ("V0", "V1", "V2")
STAGE_LABELS = {
    "V0": "第 1 关 · 原始助手",
    "V1": "第 2 关 · 教学助手",
    "V2": "第 3 关 · 技能助手",
}
STAGE_SELECTOR_LABELS = {
    "V0": "1 原始",
    "V1": "2 教学",
    "V2": "3 技能",
}
STAGE_SUMMARIES = {
    "V0": "它只有 AI 原本的能力，还没有收到你的教学要求。",
    "V1": "它会先阅读你写的教学说明书，再决定怎样帮助你。",
    "V2": "它不只会聊天，还能按需使用一套整理错题的做事步骤。",
}
STAGE_CHALLENGES = {
    "V0": "发一道题，观察它会不会直接告诉你答案。",
    "V1": "把同一道题再发一次，观察教学说明书有没有改变它的回答。",
    "V2": "请它整理一道错题，观察它会不会真的使用技能。",
}
STAGE_INTROS = {
    "V0": "你好，我是还没有经过特别训练的原始助手。你可以先发一道题试试我。",
    "V1": "你好，我会先阅读你的教学说明书。你可以用刚才的问题测试我。",
    "V2": "你好，我除了会聊天，还能按需使用整理错题技能。你想让我做什么？",
}
STAGE_CHAT_PLACEHOLDERS = {
    "V0": "和原始助手聊聊，按发送键后它才会回答",
    "V1": "把问题发给教学助手",
    "V2": "把问题或整理要求发给技能助手",
}
TRAINING_COPY = {
    "V1": {
        "expander": "训练教学助手",
        "heading": "给助手一张教学说明书",
        "explanation": (
            "Prompt 就像写给新老师的说明书。"
            "你可以告诉它扮演什么角色、怎样提问，以及哪些事不能做。"
        ),
        "hint": "可以试着写清楚：不要直接给答案，一次只问一个问题。",
        "editor_label": "写给教学助手的说明书",
    },
    "V2": {
        "expander": "训练技能助手",
        "heading": "给助手一份做事步骤",
        "explanation": (
            "Skill 像一张任务卡。"
            "它告诉助手什么时候使用这项能力，以及完成任务时要按什么步骤做。"
        ),
        "hint": "可以找到整理错题的步骤，试着调整其中一句要求。",
        "editor_label": "写给技能助手的做事步骤",
    },
}
TOOL_DESCRIPTIONS = {
    "load_skill": "读取了整理错题的方法",
    "load_mistake_file": "读取了等待整理的错题",
    "save_mistake": "保存了正式错题记录",
}


def _load_artifact(stage: str) -> None:
    """把一个真实课程文件读入当前浏览器会话。"""

    try:
        artifact = get_lesson_artifact(stage)
    except LessonArtifactError as exc:
        st.session_state.lesson1_artifact_errors[stage] = str(exc)
        st.session_state.lesson1_artifacts.pop(stage, None)
        return

    st.session_state.lesson1_artifact_errors.pop(stage, None)
    st.session_state.lesson1_editor_errors.pop(stage, None)
    st.session_state.lesson1_editor_notices.pop(stage, None)
    if artifact is None:
        st.session_state.lesson1_artifacts.pop(stage, None)
        return
    st.session_state.lesson1_artifacts[stage] = artifact
    st.session_state[f"lesson1_editor_{stage.lower()}"] = artifact["content"]


def initialize_session_state() -> None:
    """集中初始化第一课所需的全部 Session State。"""

    st.session_state.setdefault("lesson1_stage", "V0")
    st.session_state.setdefault(
        "lesson1_histories",
        {stage: [] for stage in STAGES},
    )
    st.session_state.setdefault(
        "lesson1_last_results",
        {stage: None for stage in STAGES},
    )
    st.session_state.setdefault(
        "lesson1_last_errors",
        {stage: None for stage in STAGES},
    )
    st.session_state.setdefault(
        "lesson1_last_attempts",
        {stage: None for stage in STAGES},
    )
    st.session_state.setdefault("lesson1_progress_error", None)
    st.session_state.setdefault("lesson1_artifacts", {})
    st.session_state.setdefault("lesson1_artifact_errors", {})
    st.session_state.setdefault("lesson1_editor_errors", {})
    st.session_state.setdefault("lesson1_editor_notices", {})
    st.session_state.setdefault("lesson1_last_changes", {})

    if "progress" not in st.session_state:
        try:
            st.session_state.progress = load_progress()
            st.session_state.progress_error = None
        except ProgressDataError as exc:
            st.session_state.progress = default_progress()
            st.session_state.progress_error = str(exc)

    for editable_stage in ("V1", "V2"):
        if (
            editable_stage not in st.session_state.lesson1_artifacts
            and editable_stage not in st.session_state.lesson1_artifact_errors
        ):
            _load_artifact(editable_stage)


def _preview_diff(path: str, before: str, after: str) -> str:
    """生成编辑器当前内容相对磁盘快照的统一差异。"""

    return "\n".join(
        unified_diff(
            before.splitlines(),
            after.strip().splitlines(),
            fromfile=f"{path}（保存的版本）",
            tofile=f"{path}（正在编辑）",
            lineterm="",
        )
    )


def _record_success(stage: str, message: str, result: dict) -> None:
    """只在模型成功返回后追加成对历史并保存课程进度。"""

    st.session_state.lesson1_histories[stage].extend(
        [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["text"]},
        ]
    )
    if st.session_state.get("progress_error"):
        st.session_state.lesson1_progress_error = (
            "原课程进度文件无法读取，因此本次没有覆盖它。"
            f"原因：{st.session_state.progress_error}"
        )
        return

    try:
        updated = complete_module(st.session_state.progress, stage.lower())
        st.session_state.progress = save_progress(updated)
        st.session_state.lesson1_progress_error = None
    except ProgressDataError as exc:
        st.session_state.lesson1_progress_error = str(exc)


def _latest_assistant_reply(history: list[dict]) -> str | None:
    """返回某一关最后一次成功回答。"""

    for message in reversed(history):
        if message["role"] == "assistant":
            return message["content"]
    return None


@contextmanager
def _chat_bubble(role: str) -> Iterator[None]:
    """用原生布局让助手消息靠左、学生消息靠右。"""

    is_user = role == "user"
    alignment = "right" if is_user else "left"
    avatar = ":material/person:" if is_user else ":material/smart_toy:"
    with st.container(horizontal_alignment=alignment, gap=None):
        with st.chat_message(role, avatar=avatar, width="content"):
            yield


st.set_page_config(
    page_title="第一课 - 训练你的学习助手",
    page_icon=":material/school:",
    layout="centered",
)

initialize_session_state()

completed_modules = set(st.session_state.progress["completed_modules"])
completed_count = sum(stage_name.lower() in completed_modules for stage_name in STAGES)

with st.sidebar:
    st.header("第一课")
    st.caption("训练你的学习助手")
    st.write("选择一个关卡，中间的聊天窗口会切换到对应的助手。")

    stage = st.segmented_control(
        "选择一关",
        STAGES,
        format_func=lambda value: STAGE_SELECTOR_LABELS[value],
        required=True,
        key="lesson1_stage",
        width="stretch",
        persist_state="session",
    )
    stage = stage or "V0"

    st.markdown(f"**{STAGE_LABELS[stage]}**")
    st.caption(STAGE_SUMMARIES[stage])
    st.caption(f"本关挑战：{STAGE_CHALLENGES[stage]}")

    artifact = st.session_state.lesson1_artifacts.get(stage)
    artifact_error = st.session_state.lesson1_artifact_errors.get(stage)

    if stage == "V0":
        st.caption("这一关先不训练助手，把它当作后面两关的对照组。")
    else:
        training_copy = TRAINING_COPY[stage]
        with st.expander(
            training_copy["expander"],
            icon=":material/tune:",
        ):
            st.markdown(f"**{training_copy['heading']}**")
            st.write(training_copy["explanation"])
            st.info(training_copy["hint"], icon=":material/lightbulb:")

            if artifact_error:
                st.error(artifact_error)
                reload_after_error = st.button(
                    "重新读取",
                    icon=":material/refresh:",
                    key=f"lesson1_reload_error_{stage}",
                )
                if reload_after_error:
                    _load_artifact(stage)
                    st.rerun()
            elif artifact is not None:
                editor_error = st.session_state.lesson1_editor_errors.get(stage)
                editor_notice = st.session_state.lesson1_editor_notices.get(stage)
                if editor_error:
                    st.error(editor_error)
                if editor_notice:
                    st.success(editor_notice, icon=":material/check_circle:")

                edited_content = st.text_area(
                    training_copy["editor_label"],
                    key=f"lesson1_editor_{stage.lower()}",
                    height=240,
                    max_chars=256 * 1024,
                    help=(
                        "Markdown 是带有标题和列表的普通文字。"
                        "你只需要修改看得懂的句子。"
                    ),
                    persist_state="session",
                )

                with st.container(horizontal=True):
                    save_clicked = st.button(
                        "保存给助手",
                        type="primary",
                        icon=":material/save:",
                        key=f"lesson1_save_{stage}",
                    )
                    reload_clicked = st.button(
                        "放弃修改",
                        icon=":material/refresh:",
                        key=f"lesson1_reload_{stage}",
                        help="重新读取上次保存的内容。",
                    )

                st.caption("保存只更新助手的说明，不发送消息，也不调用 AI。")

                if save_clicked:
                    try:
                        change = save_lesson_artifact(
                            stage,
                            edited_content,
                            expected_digest=artifact["digest"],
                        )
                    except LessonArtifactError as exc:
                        st.session_state.lesson1_editor_errors[stage] = str(exc)
                        st.session_state.lesson1_editor_notices.pop(stage, None)
                    else:
                        st.session_state.lesson1_artifacts[stage] = change["after"]
                        st.session_state.lesson1_last_changes[stage] = change
                        st.session_state.lesson1_editor_errors.pop(stage, None)
                        if change["changed"]:
                            notice = "已保存。下次发送消息时，新说明就会生效。"
                        else:
                            notice = "内容没有变化，不需要重复保存。"
                        st.session_state.lesson1_editor_notices[stage] = notice
                    st.rerun()

                if reload_clicked:
                    _load_artifact(stage)
                    st.rerun()

                pending_diff = _preview_diff(
                    artifact["path"],
                    artifact["content"],
                    edited_content or "",
                )
                st.markdown("**看看修改前后有什么不同**")
                st.caption("- 是原来的文字，+ 是你新写的文字。")
                if pending_diff:
                    st.code(
                        pending_diff,
                        language="diff",
                        wrap_lines=True,
                        height=180,
                    )
                else:
                    st.caption("现在的内容和上次保存的版本相同。")

                last_change = st.session_state.lesson1_last_changes.get(stage)
                if last_change and last_change["changed"]:
                    st.caption("上一次修改已经保存成功。")
                st.caption(f"幕后文件：`{artifact['path']}`")

    st.progress(
        completed_count / len(STAGES),
        text=f"第一课已体验 {completed_count}/3 关",
    )

    if st.session_state.lesson1_progress_error:
        st.warning(
            "助手已经回答，但课程进度没有保存。"
            f"原因：{st.session_state.lesson1_progress_error}"
        )

    last_result = st.session_state.lesson1_last_results[stage]
    if last_result and last_result["tool_calls"]:
        with st.expander("这次助手做了什么", icon=":material/bolt:"):
            for call in last_result["tool_calls"]:
                name = call.get("name", "unknown")
                description = TOOL_DESCRIPTIONS.get(name, "使用了一项工具")
                st.markdown(f"- {description}")
            with st.expander("查看开发者信息", icon=":material/code:"):
                st.caption("这里记录了工具名称和参数，现在看不懂也没关系。")
                for index, call in enumerate(last_result["tool_calls"], start=1):
                    st.markdown(f"**{index}. `{call.get('name', 'unknown')}`**")
                    st.json(call.get("args", {}), expanded=False)

    with st.expander("比较三个助手的回答", icon=":material/compare_arrows:"):
        st.write("把同一个问题发给不同关卡，再比较它们最后一次回答的区别。")
        for comparison_stage in STAGES:
            st.markdown(f"**{STAGE_LABELS[comparison_stage]}**")
            latest_reply = _latest_assistant_reply(
                st.session_state.lesson1_histories[comparison_stage]
            )
            if latest_reply:
                st.markdown(latest_reply)
            else:
                st.caption("还没有成功回答过。")

history = st.session_state.lesson1_histories[stage]

with _chat_bubble("assistant"):
    st.write(STAGE_INTROS[stage])

for chat_message in history:
    with _chat_bubble(chat_message["role"]):
        st.write(chat_message["content"])

last_attempt = st.session_state.lesson1_last_attempts[stage]
last_error = st.session_state.lesson1_last_errors[stage]
if last_error and last_attempt:
    with _chat_bubble("user"):
        st.write(last_attempt)
    with _chat_bubble("assistant"):
        st.error(f"这条消息没有发送成功。{last_error}")

chat_disabled = bool(artifact_error)
chat_placeholder = STAGE_CHAT_PLACEHOLDERS[stage]
if chat_disabled:
    chat_placeholder = "先重新读取助手的说明，再继续聊天"

message = st.chat_input(
    chat_placeholder,
    key=f"lesson1_chat_{stage.lower()}",
    max_chars=8_000,
    disabled=chat_disabled,
    submit_mode="disable",
)

if message is not None:
    clean_message = message.strip()
    if not clean_message:
        st.session_state.lesson1_last_attempts[stage] = message
        st.session_state.lesson1_last_errors[stage] = "请输入内容后再发送。"
        st.rerun()

    st.session_state.lesson1_last_attempts[stage] = clean_message
    st.session_state.lesson1_last_errors[stage] = None
    with _chat_bubble("user"):
        st.write(clean_message)
    with _chat_bubble("assistant"):
        with st.spinner("助手正在思考…", show_time=True):
            result = invoke(
                stage,
                clean_message,
                history=[item.copy() for item in history],
            )
        st.session_state.lesson1_last_results[stage] = result
        if result["error"]:
            st.session_state.lesson1_last_errors[stage] = result["error"]
            st.error(f"这条消息没有发送成功。{result['error']}")
        else:
            st.write(result["text"])
            st.session_state.lesson1_last_attempts[stage] = None
            _record_success(stage, clean_message, result)
    st.rerun()
