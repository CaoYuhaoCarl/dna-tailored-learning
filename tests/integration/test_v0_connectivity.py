from base64 import b64decode
import os

from dotenv import dotenv_values
import pytest

from src.facade import invoke
from src.chat_submission import create_chat_attachment
import src.model as model_module


_TINY_PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.mark.integration
def test_v0_connects_to_selected_model() -> None:
    if os.getenv("RUN_MODEL_INTEGRATION") != "1":
        pytest.skip("设置 RUN_MODEL_INTEGRATION=1 后才调用真实模型。")

    result = invoke("V0", "请只回复：V0 连接成功")

    assert result["error"] is None, result["error"]
    assert result["text"].strip()


@pytest.mark.integration
def test_deepseek_v4_flash_vision_accepts_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.getenv("RUN_DEEPSEEK_INTEGRATION") != "1":
        pytest.skip("设置 RUN_DEEPSEEK_INTEGRATION=1 后才调用真实 DeepSeek。")

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        api_key = str(
            dotenv_values(model_module.ENV_FILE).get("DEEPSEEK_API_KEY") or ""
        ).strip()
    if not api_key:
        pytest.fail("真实 DeepSeek 测试需要 DEEPSEEK_API_KEY。", pytrace=False)

    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", api_key)
    del api_key

    content = None
    try:
        llm = model_module.get_llm()
        assert model_module.get_model_name(llm) == (
            "deepseek-v4-flash-vision-exp"
        )
        attachment = create_chat_attachment(
            name="integration.png",
            media_type="image/png",
            data=_TINY_PNG,
        )
        result = invoke(
            "V0",
            "请用一句话描述这张图片，并且不要返回图片编码。",
            attachment=attachment,
        )
        content = result["text"]
        if result["error"]:
            raise RuntimeError(result["error"])
    except Exception as exc:
        error_type = type(exc).__name__
    else:
        error_type = None

    if error_type is not None:
        pytest.fail(
            "DeepSeek V4 Flash Vision 图片连通性检查失败，"
            f"异常类型：{error_type}。",
            pytrace=False,
        )

    if not content or (isinstance(content, str) and not content.strip()):
        pytest.fail("DeepSeek V4 Flash Vision 返回了空内容。", pytrace=False)
