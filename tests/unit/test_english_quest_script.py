import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "student"
    / "skill"
    / "english-quest"
    / "scripts"
    / "quest_state.py"
)


def _derive(history: list[dict]) -> dict:
    namespace = runpy.run_path(str(SCRIPT_PATH))
    return namespace["derive_quest_state"](history)


def test_quest_state_script_extracts_latest_visible_status() -> None:
    history = [
        {"role": "assistant", "content": "关卡：1/5 生命：❤️❤️❤️ 经验：0 XP"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "关卡：2/5 生命：❤️❤️❤️ 经验：10 XP"},
    ]

    assert _derive(history) == {
        "level": 2,
        "hearts": 3,
        "xp": 10,
        "complete": False,
    }


def test_quest_state_script_handles_retry_and_final_report() -> None:
    retry_history = [
        {"role": "assistant", "content": "第 3 关\n生命：2\nXP：20"},
    ]
    completed_history = retry_history + [
        {
            "role": "assistant",
            "content": "任务报告\n最终 XP：40\n剩余生命：2",
        }
    ]

    assert _derive(retry_history) == {
        "level": 3,
        "hearts": 2,
        "xp": 20,
        "complete": False,
    }
    assert _derive(completed_history) == {
        "level": 5,
        "hearts": 2,
        "xp": 40,
        "complete": True,
    }


def test_quest_state_script_accepts_natural_status_without_colons() -> None:
    history = [
        {
            "role": "assistant",
            "content": "关卡：1/5 生命：❤️❤️❤️ 经验：0 XP",
        },
        {
            "role": "assistant",
            "content": "关卡 1/5 生命 ❤️❤️❤️ 获得 10 XP",
        },
        {
            "role": "assistant",
            "content": "关卡 2/5 生命 ❤️❤️ 获得 10 XP",
        },
    ]

    assert _derive(history) == {
        "level": 2,
        "hearts": 2,
        "xp": 20,
        "complete": False,
    }
