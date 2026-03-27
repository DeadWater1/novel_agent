from __future__ import annotations

from novel_agent.backends.compression import _forced_think_close_ids
from novel_agent.utils import extract_answer_text, extract_json_object, split_think_and_answer


class FakeTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        if token == "</think>":
            return 999
        return -1

    def decode(self, output_ids, skip_special_tokens=True):
        mapping = {
            1: "<think>",
            2: "分析用户是否要压缩",
            999: "</think>",
            3: "这是压缩后的结果",
        }
        return "".join(mapping[item] for item in output_ids)


class TextFirstTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return -1

    def decode(self, output_ids, skip_special_tokens=True):
        return "<think>先分析人物与战场线索</think>这是压缩后的结果"


def test_extract_answer_text_removes_think_block():
    text = "<think>内部思考</think>这是最终回答"
    assert extract_answer_text(text) == "这是最终回答"


def test_split_think_and_answer_matches_qwen_style():
    tokenizer = FakeTokenizer()
    thinking, answer = split_think_and_answer(tokenizer, [1, 2, 999, 3])
    assert thinking == "分析用户是否要压缩"
    assert answer == "这是压缩后的结果"


def test_split_think_and_answer_prefers_text_tags_before_token_boundary():
    tokenizer = TextFirstTokenizer()
    thinking, answer = split_think_and_answer(tokenizer, [42, 43, 44])
    assert thinking == "先分析人物与战场线索"
    assert answer == "这是压缩后的结果"


def test_extract_json_object_accepts_wrapped_text():
    payload = extract_json_object("前缀 {\"domain\":\"novel\",\"action\":\"direct_reply\"} 后缀")
    assert payload["domain"] == "novel"
    assert payload["action"] == "direct_reply"


def test_forced_think_close_ids_include_think_end_token():
    tokenizer = FakeTokenizer()
    closing_ids = _forced_think_close_ids(tokenizer, 999)
    assert 999 in closing_ids
