from novel_agent.app import (
    PROCESSING_PLACEHOLDER,
    _append_chat_history,
    _append_pending_chat_history,
    _finalize_chat_history,
    _render_backend_status,
    _render_chat_html,
    _render_hero,
    _render_loop_trace,
)


def test_append_chat_history_appends_two_messages():
    history = []
    history = _append_chat_history(history, "用户问题", "助手回答")
    assert history == [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": "助手回答"},
    ]


def test_append_pending_chat_history_adds_processing_placeholder():
    history = []
    history = _append_pending_chat_history(history, "用户问题")
    assert history == [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": PROCESSING_PLACEHOLDER},
    ]


def test_finalize_chat_history_replaces_processing_placeholder():
    history = [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": PROCESSING_PLACEHOLDER},
    ]
    updated = _finalize_chat_history(history, "用户问题", "最终回复")
    assert updated == [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": "最终回复"},
    ]


def test_render_chat_html_contains_both_messages():
    history = [
        {"role": "user", "content": "用户问题"},
        {"role": "assistant", "content": "助手回答"},
    ]
    html = _render_chat_html(history)
    assert "用户问题" in html
    assert "助手回答" in html


def test_render_hero_only_shows_app_title():
    html = _render_hero("Novel Agent V3")
    assert "Novel Agent V3" in html
    assert "Closed-Domain Novel Agent" not in html
    assert "Markdown Memory" not in html


def test_render_backend_status_wraps_body_with_runtime_scroll_container():
    html = _render_backend_status(
        {
            "decision_backend": {"ok": True, "detail": "ready"},
            "compression_backend": {"ok": True, "detail": "ready"},
        }
    )
    assert "panel-body-scroll runtime-scroll" in html
    assert "status-list" in html


def test_render_loop_trace_wraps_body_with_loop_scroll_container():
    html = _render_loop_trace([])
    assert "panel-body-scroll loop-scroll" in html
    assert "empty-note" in html
