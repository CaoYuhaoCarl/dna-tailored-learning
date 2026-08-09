import os

import pytest

from src.facade import invoke


@pytest.mark.integration
def test_v0_connects_to_deepseek() -> None:
    if os.getenv("RUN_DEEPSEEK_INTEGRATION") != "1":
        pytest.skip("设置 RUN_DEEPSEEK_INTEGRATION=1 后才调用真实 DeepSeek。")

    result = invoke("V0", "请只回复：V0 连接成功")

    assert result["error"] is None, result["error"]
    assert result["text"].strip()
