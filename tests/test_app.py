from novel_agent.app import _append_chat_history, _render_chat_html


def test_append_chat_history_appends_two_messages():
    history = []
    history = _append_chat_history(history, "用户问题", "助手回答")
    assert history == [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": "助手回答"},
    ]


def test_render_chat_html_contains_both_messages():
    history = [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": "助手回答"},
    ]
    html = _render_chat_html(history)
    assert "用户问题" in html
    assert "助手回答" in html
