from novel_agent.app import (
    PROCESSING_PLACEHOLDER,
    _append_chat_history,
    _append_pending_chat_history,
    _render_decision_output,
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
    html = _render_hero("Novel Agent V4")
    assert "Novel Agent V4" in html
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


def test_render_loop_trace_shows_plan_events():
    html = _render_loop_trace(
        [
            {"event_type": "plan_created", "payload": {"user_goal": "测试"}},
            {"event_type": "plan_step_completed", "step_index": 1, "payload": {"goal": "先检索"}},
            {"event_type": "plan_updated", "step_index": 2, "payload": {"user_goal": "重规划"}},
        ]
    )
    assert "Plan Created" in html
    assert "Step Completed" in html
    assert "Plan Updated" in html


def test_render_decision_output_wraps_body_with_decision_scroll_container():
    html = _render_decision_output('{"steps":[{"step_index":1}]}', '{"action":"direct_reply"}', '{"verdict":"retry"}')
    assert "panel-body-scroll decision-scroll" in html
    assert "Planner Output" in html
    assert "Decision Output" in html
    assert "Review Output" in html
