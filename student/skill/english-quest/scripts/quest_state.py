"""从 english-quest 对话中提取供互动页面展示的游戏状态。"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from typing import TypedDict


class QuestState(TypedDict):
    """页面展示所需的确定性状态。"""

    level: int
    hearts: int
    xp: int
    complete: bool


_LEVEL_PATTERNS = (
    re.compile(r"关卡\s*[：:]?\s*(\d+)\s*/\s*5"),
    re.compile(r"第\s*(\d+)\s*关"),
)
_HEARTS_NUMBER_PATTERN = re.compile(r"(?:生命|剩余生命)\s*[：:]?\s*(\d+)")
_HEARTS_EMOJI_PATTERN = re.compile(r"生命\s*[：:]?\s*((?:❤️|❤|♥️?)+)")
_EXPLICIT_XP_PATTERNS = (
    re.compile(r"经验\s*[：:]?\s*(\d+)\s*XP", re.IGNORECASE),
    re.compile(r"(?:最终\s*)?XP\s*[：:]?\s*(\d+)", re.IGNORECASE),
)
_XP_GAIN_PATTERN = re.compile(r"获得\s*(\d+)\s*XP", re.IGNORECASE)
_COMPLETE_MARKERS = ("任务报告", "任务完成", "闯关完成")


def _assistant_texts(history: Sequence[dict[str, object]]) -> list[str]:
    return [
        str(item.get("content", ""))
        for item in history
        if item.get("role") == "assistant" and item.get("content")
    ]


def _latest_match(
    texts: Sequence[str],
    patterns: Sequence[re.Pattern[str]],
) -> int | None:
    for text in reversed(texts):
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return int(match.group(1))
    return None


def _latest_hearts(texts: Sequence[str]) -> int | None:
    for text in reversed(texts):
        emoji_match = _HEARTS_EMOJI_PATTERN.search(text)
        if emoji_match:
            return len(re.findall(r"❤️|❤|♥️?", emoji_match.group(1)))
        number_match = _HEARTS_NUMBER_PATTERN.search(text)
        if number_match:
            return int(number_match.group(1))
    return None


def _latest_xp(texts: Sequence[str]) -> int:
    for index in range(len(texts) - 1, -1, -1):
        for pattern in _EXPLICIT_XP_PATTERNS:
            match = pattern.search(texts[index])
            if match:
                later_gains = sum(
                    int(gain.group(1))
                    for text in texts[index + 1 :]
                    for gain in _XP_GAIN_PATTERN.finditer(text)
                )
                return int(match.group(1)) + later_gains
    return sum(
        int(match.group(1))
        for text in texts
        for match in _XP_GAIN_PATTERN.finditer(text)
    )


def derive_quest_state(history: Sequence[dict[str, object]]) -> QuestState:
    """读取对话，不猜测答题对错，只返回可验证的展示状态。"""

    texts = _assistant_texts(history)
    latest_text = texts[-1] if texts else ""
    complete = any(marker in latest_text for marker in _COMPLETE_MARKERS)
    level = _latest_match(texts, _LEVEL_PATTERNS)
    hearts = _latest_hearts(texts)
    xp = _latest_xp(texts)
    return {
        "level": 5 if complete else max(0, min(level or 0, 5)),
        "hearts": max(0, min(hearts if hearts is not None else 3, 3)),
        "xp": max(0, xp),
        "complete": complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="提取英语闯关页面状态")
    parser.add_argument("history_json", help="包含对话历史的 JSON 文件")
    args = parser.parse_args()
    with open(args.history_json, encoding="utf-8") as history_file:
        history = json.load(history_file)
    print(json.dumps(derive_quest_state(history), ensure_ascii=False))


if __name__ == "__main__":
    main()
