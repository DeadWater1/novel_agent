from __future__ import annotations

import json
import random
from typing import Any


THINK_END_TOKEN_FALLBACK = 151668


def resolve_seed(seed: int | str | None) -> int:
    try:
        parsed = int(seed) if seed is not None else -1
    except (TypeError, ValueError):
        parsed = -1
    if parsed == -1:
        return random.randint(0, 2**31 - 1)
    return parsed


def get_think_end_token_id(tokenizer: Any) -> int:
    token_id = None
    try:
        token_id = tokenizer.convert_tokens_to_ids("</think>")
    except Exception:
        token_id = None
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        token_id = THINK_END_TOKEN_FALLBACK
    return int(token_id)


def extract_think_text(thinking_text: str) -> str:
    if not thinking_text:
        return ""
    if "<think>" in thinking_text and "</think>" in thinking_text:
        head = thinking_text.split("<think>", 1)[-1]
        body = head.split("</think>", 1)[0]
        return body.strip()
    return thinking_text.strip()


def extract_answer_text(output_text: str) -> str:
    if not output_text:
        return ""
    text = str(output_text).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text and "</think>" in text:
        prefix, _, _ = text.partition("<think>")
        text = prefix + text.split("</think>", 1)[-1]
    if text.startswith("<think>"):
        text = text.split("<think>", 1)[-1].strip()
    return text.strip()


def split_think_and_answer(tokenizer: Any, output_ids: list[int]) -> tuple[str, str]:
    end_id = get_think_end_token_id(tokenizer)
    try:
        index = len(output_ids) - output_ids[::-1].index(end_id)
    except ValueError:
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return "", extract_answer_text(decoded)
    thinking_raw = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return extract_think_text(thinking_raw), extract_answer_text(answer)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])
