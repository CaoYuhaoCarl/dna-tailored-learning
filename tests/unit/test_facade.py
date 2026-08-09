import src.facade as facade_module


def test_invoke_rejects_unavailable_stage() -> None:
    result = facade_module.invoke("V1", "测试")

    assert result["stage"] == "V1"
    assert result["error"] == "当前版本仅支持 V0，收到的阶段为 V1。"
