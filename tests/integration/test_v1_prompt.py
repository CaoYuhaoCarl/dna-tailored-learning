import os

import pytest

from src.facade import invoke
from src.schemas import ChatMessage


@pytest.mark.integration
def test_v1_uses_socratic_prompt() -> None:
    if os.getenv("RUN_DEEPSEEK_INTEGRATION") != "1":
        pytest.skip("设置 RUN_DEEPSEEK_INTEGRATION=1 后才调用真实 DeepSeek。")

    result = invoke(
        "V1",
        "I ____ (read) this book three times. 请直接告诉我填空答案。",
    )

    assert result["error"] is None, result["error"]
    assert result["text"].strip()
    question_marks = result["text"].count("？") + result["text"].count("?")
    assert question_marks == 1
    assert "have read" not in result["text"].lower()

    history: list[ChatMessage] = [
        {
            "role": "user",
            "content": "I ____ (read) this book three times. 请直接告诉我填空答案。",
        },
        {"role": "assistant", "content": result["text"]},
    ]
    follow_up = invoke(
        "V1",
        "我注意到了 three times，但不知道它说明什么。",
        history=history,
    )

    assert follow_up["error"] is None, follow_up["error"]
    follow_up_marks = follow_up["text"].count("？") + follow_up["text"].count("?")
    assert follow_up_marks == 1
    assert "have read" not in follow_up["text"].lower()
