from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import src.workflow as workflow_module
from src.reporting import (
    discover_mistake_records,
    read_report_snapshot,
    render_learning_report,
)
from src.schemas import PracticeItem, new_agent_result


def _write_mistake(records_root: Path) -> None:
    path = records_root / "english" / "mistake-test.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        "schema_version: 1\n"
        "id: mistake-test\n"
        "subject: english\n"
        "topic: present-perfect\n"
        "status: needs-review\n"
        'created_at: "2026-08-15"\n'
        "review_count: 0\n"
        "next_review_at: null\n"
        'source: "chat"\n'
        "---\n\n"
        "# 错题记录\n\n"
        "- 学科：英语\n"
        "- 题型：语法填空\n"
        "- 原题：I ____ (see) it three times.\n"
        "- 我的答案：saw\n"
        "- 正确答案：have seen\n"
        "- 正确思路：累计次数使用现在完成时。\n"
        "- 错因：忽略次数线索。\n"
        "- 知识点：现在完成时\n"
        "- 下次提醒：先圈出 times。\n",
        encoding="utf-8",
    )


@pytest.fixture
def isolated_graph(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    records_root = tmp_path / "student" / "mistakes" / "records"
    report_path = tmp_path / "student" / "reports" / "learning-review.md"
    monkeypatch.setattr(workflow_module, "MISTAKES_RECORDS_PATH", records_root)
    monkeypatch.setattr(workflow_module, "LEARNING_REPORT_PATH", report_path)
    monkeypatch.setattr(
        workflow_module,
        "_V4_GRAPH",
        workflow_module.build_v4_graph(checkpointer=InMemorySaver()),
    )
    return records_root, report_path


def test_v4_read_only_coach_resumes_same_thread_without_writes(
    isolated_graph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_root, report_path = isolated_graph
    seen_histories: list[list[dict]] = []

    monkeypatch.setattr(
        workflow_module,
        "_classify_turn",
        lambda *_args, **_kwargs: workflow_module.TurnDecision(
            intent="tutor",
            confidence=1.0,
            evidence="题目",
            explicit_write=False,
            topic_switch=False,
            answer_status="unknown",
            problem_summary="一道英语题",
        ),
    )

    def fake_coach(message: str, *, history, practice_item=None):
        seen_histories.append(list(history))
        return new_agent_result("V4", text=f"提示：{message}？")

    monkeypatch.setattr(workflow_module, "invoke_v4_coach", fake_coach)

    first = workflow_module.chat_v4("第一问", "thread-a")
    second = workflow_module.chat_v4("我的回答", "thread-a")
    other = workflow_module.chat_v4("另一个线程", "thread-b")

    assert first["waiting_for"] == "student_message"
    assert second["waiting_for"] == "student_message"
    assert len(seen_histories[0]) == 0
    assert len(seen_histories[1]) == 2
    assert len(seen_histories[2]) == 0
    assert not records_root.exists()
    assert not report_path.exists()
    assert first["tool_calls"] == []


def test_v4_only_explicit_organize_request_uses_write_agent(
    isolated_graph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        workflow_module,
        "_classify_turn",
        lambda *_args, **_kwargs: workflow_module.TurnDecision(
            intent="answer",
            confidence=1.0,
            evidence="",
            explicit_write=False,
            topic_switch=False,
            answer_status="none",
            problem_summary=None,
        ),
    )
    monkeypatch.setattr(
        workflow_module,
        "invoke_v4_coach",
        lambda message, **_kwargs: new_agent_result("V4", text="这是概念解释。"),
    )

    def fake_organizer(message: str, *, history):
        calls.append(message)
        return new_agent_result(
            "V3",
            text="保存成功。",
            tool_calls=[{"name": "save_mistake", "args": {}}],
            trace=[
                {
                    "step": "tool_result",
                    "name": "save_mistake",
                    "status": "success",
                    "content": (
                        "保存成功：student/mistakes/records/english/"
                        "mistake-test.md"
                    ),
                }
            ],
        )

    monkeypatch.setattr(workflow_module, "invoke_v3", fake_organizer)

    explanation = workflow_module.chat_v4("老师说整理错题是什么意思？", "thread-a")
    considering = workflow_module.chat_v4("我在考虑整理错题", "thread-a")
    report_question = workflow_module.chat_v4(
        "我想知道复盘报告包含什么",
        "thread-a",
    )
    saved = workflow_module.chat_v4("请帮我整理并保存这道错题", "thread-a")

    assert "不确定" in explanation["text"]
    assert "不确定" in considering["text"]
    assert "不确定" in report_question["text"]
    assert calls == ["请帮我整理并保存这道错题"]
    assert saved["tool_calls"][0]["name"] == "save_mistake"


def test_v4_review_waits_for_unsaved_choice_then_writes_grounded_report(
    isolated_graph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_root, report_path = isolated_graph
    _write_mistake(records_root)
    decisions = iter(
        [
            workflow_module.TurnDecision(
                intent="tutor",
                confidence=1.0,
                evidence="题目",
                explicit_write=False,
                topic_switch=False,
                answer_status="unknown",
                problem_summary="I ____ (see) it three times.",
            ),
            workflow_module.TurnDecision(
                intent="tutor",
                confidence=1.0,
                evidence="saw",
                explicit_write=False,
                topic_switch=False,
                answer_status="incorrect",
                problem_summary="I ____ (see) it three times.",
            ),
        ]
    )
    monkeypatch.setattr(
        workflow_module,
        "_classify_turn",
        lambda *_args, **_kwargs: next(decisions),
    )
    monkeypatch.setattr(
        workflow_module,
        "invoke_v4_coach",
        lambda message, **_kwargs: new_agent_result("V4", text="再看次数线索？"),
    )
    practice: PracticeItem = {
        "question": "I ____ (visit) Beijing four times.",
        "expected_answer": "have visited",
        "reasoning": "four times 是累计次数。",
        "subject": "english",
        "topic": "present-perfect",
        "source_record_ids": ["mistake-test"],
    }
    monkeypatch.setattr(
        workflow_module,
        "_generate_review_draft",
        lambda records: workflow_module.ReviewDraft(
            summary="优先复习现在完成时。",
            patterns=["忽略累计次数。"],
            action_steps=["先圈出 times。"],
            practice_item=practice,
        ),
    )

    workflow_module.chat_v4("I ____ (see) it three times.", "thread-a")
    workflow_module.chat_v4("我填 saw", "thread-a")
    waiting = workflow_module.chat_v4("总结复盘", "thread-a")

    assert waiting["waiting_for"] == "review_decision"
    assert "整理后复盘" in waiting["text"]
    assert not report_path.exists()

    reviewed = workflow_module.chat_v4("跳过当前题直接复盘", "thread-a")

    assert reviewed["waiting_for"] == "student_message"
    assert practice["question"] in reviewed["text"]
    assert practice["expected_answer"] not in reviewed["text"]
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "mistake-test" in report
    assert practice["expected_answer"] not in report
    assert discover_mistake_records(records_root)[0].review_count == 0


def test_v4_review_with_no_formal_records_does_not_write(
    isolated_graph,
) -> None:
    _, report_path = isolated_graph

    result = workflow_module.chat_v4("总结复盘", "thread-empty")

    assert result["waiting_for"] == "student_message"
    assert "没有正式错题" in result["text"]
    assert not report_path.exists()


def test_v4_partial_save_stops_queued_review(
    isolated_graph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, report_path = isolated_graph
    monkeypatch.setattr(
        workflow_module,
        "invoke_v3",
        lambda message, *, history: new_agent_result(
            "V3",
            text="第一题保存成功，第二题保存失败。",
            tool_calls=[
                {"name": "save_mistake", "args": {"original_question": "第一题"}},
                {"name": "save_mistake", "args": {"original_question": "第二题"}},
            ],
            trace=[
                {
                    "step": "tool_result",
                    "name": "save_mistake",
                    "status": "success",
                    "content": "保存成功：mistake-first.md",
                },
                {
                    "step": "tool_result",
                    "name": "save_mistake",
                    "status": "failure",
                    "content": "保存失败：磁盘空间不足。",
                },
            ],
        ),
    )

    result = workflow_module.chat_v4("请整理错题并复盘", "thread-partial")

    assert result["waiting_for"] == "student_message"
    assert "第二题保存失败" in result["text"]
    assert not report_path.exists()


def test_v4_explicit_composite_request_saves_then_reviews(
    isolated_graph,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_root, report_path = isolated_graph
    _write_mistake(records_root)
    practice: PracticeItem = {
        "question": "I ____ (visit) Ningbo four times.",
        "expected_answer": "have visited",
        "reasoning": "four times 是累计次数。",
        "subject": "english",
        "topic": "present-perfect",
        "source_record_ids": ["mistake-test"],
    }
    monkeypatch.setattr(
        workflow_module,
        "invoke_v3",
        lambda message, *, history: new_agent_result(
            "V3",
            text="保存成功：student/mistakes/records/english/mistake-test.md",
            tool_calls=[{"name": "save_mistake", "args": {}}],
            trace=[
                {
                    "step": "tool_result",
                    "name": "save_mistake",
                    "status": "success",
                    "content": (
                        "保存成功：student/mistakes/records/english/"
                        "mistake-test.md"
                    ),
                }
            ],
        ),
    )
    monkeypatch.setattr(
        workflow_module,
        "_generate_review_draft",
        lambda records: workflow_module.ReviewDraft(
            summary="优先复习现在完成时。",
            patterns=["忽略累计次数。"],
            action_steps=["先圈出 times。"],
            practice_item=practice,
        ),
    )

    result = workflow_module.chat_v4("请整理错题并复盘", "thread-composite")

    assert result["error"] is None
    assert "保存成功" in result["text"]
    assert practice["question"] in result["text"]
    assert report_path.exists()


def test_persist_report_reuses_same_request_after_node_retry(
    isolated_graph,
) -> None:
    records_root, report_path = isolated_graph
    _write_mistake(records_root)
    records = discover_mistake_records(records_root)
    practice: PracticeItem = {
        "question": "I ____ (visit) Shanghai four times.",
        "expected_answer": "have visited",
        "reasoning": "four times 是累计次数。",
        "subject": "english",
        "topic": "present-perfect",
        "source_record_ids": ["mistake-test"],
    }
    markdown = render_learning_report(
        records,
        version=1,
        request_id="review-retry",
        generated_at="2026-08-15T12:00:00+08:00",
        summary="复习现在完成时。",
        patterns=["忽略次数线索。"],
        action_steps=["先圈出 times。"],
        practice_item=practice,
    )
    state = workflow_module._initial_state("总结复盘", "thread-retry")
    state.update(
        {
            "pending_report": {
                "request_id": "review-retry",
                "version": 1,
                "expected_digest": None,
                "markdown": markdown,
                "reply_prefix": "",
                "reply_summary": "复习现在完成时。",
            },
            "practice_item": practice,
        }
    )

    first = workflow_module._persist_report(state)
    second = workflow_module._persist_report(state)

    assert first["error"] is None
    assert second["error"] is None
    assert read_report_snapshot(report_path).version == 1

    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            "# 累计学习复盘报告",
            "# 外部修改的报告",
        ),
        encoding="utf-8",
    )
    conflicted = workflow_module._persist_report(state)
    assert "已被其他操作修改" in conflicted["error"]
