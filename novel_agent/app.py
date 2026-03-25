from __future__ import annotations

import html
from typing import Any

from .backends import LocalCompressionBackend, LocalDecisionBackend
from .config import AgentConfig
from .controller import ControllerDependencies, NovelAgentController
from .heartbeat import HeartbeatManager
from .memory import SessionState, SessionStore
from .registry import build_default_registry
from .session_meta import SessionMetaStore
from .workspace import WorkspaceManager


APP_CSS = """
:root {
  --paper-1: #f7efe2;
  --paper-2: #fff8ef;
  --paper-3: rgba(255, 250, 241, 0.84);
  --ink-1: #000000;
  --ink-2: #000000;
  --line: rgba(125, 98, 70, 0.18);
  --accent: #000000;
  --accent-soft: rgba(178, 90, 44, 0.12);
  --olive: #000000;
  --olive-soft: rgba(47, 93, 81, 0.12);
  --shadow: 0 24px 60px rgba(75, 55, 33, 0.10);
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(222, 192, 138, 0.34), transparent 34%),
    radial-gradient(circle at top right, rgba(107, 147, 135, 0.18), transparent 26%),
    linear-gradient(180deg, #fbf4e8 0%, #f5ede0 100%);
  color: var(--ink-1);
  font-family: "IBM Plex Sans", "Noto Sans SC", "PingFang SC", sans-serif;
}

.gradio-container {
  max-width: 1440px !important;
  padding: 24px 20px 32px !important;
}

.block-title {
  display: none !important;
}

.hero-card,
.chat-card,
.composer-card,
.side-card,
.thinking-card {
  border: 1px solid var(--line);
  background: var(--paper-3);
  border-radius: 24px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
}

.hero-card {
  overflow: hidden;
}

.hero-inner {
  display: flex;
  justify-content: space-between;
  gap: 28px;
  padding: 28px;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.34), rgba(255, 255, 255, 0.08)),
    linear-gradient(120deg, rgba(178, 90, 44, 0.10), rgba(47, 93, 81, 0.08));
}

.hero-copy {
  max-width: 760px;
}

.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.60);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.hero-title {
  margin: 16px 0 10px;
  color: var(--ink-1);
  font-size: 44px;
  line-height: 1.04;
  font-family: "Source Han Serif SC", "Noto Serif SC", "Songti SC", Georgia, serif;
}

.hero-subtitle {
  margin: 0;
  max-width: 680px;
  color: var(--ink-2);
  font-size: 16px;
  line-height: 1.78;
}

.hero-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-content: flex-start;
}

.hero-tag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 142px;
  padding: 12px 16px;
  border-radius: 18px;
  border: 1px solid rgba(125, 98, 70, 0.16);
  background: rgba(255, 255, 255, 0.62);
  color: var(--ink-1);
  font-size: 13px;
  font-weight: 600;
}

.layout-row {
  align-items: stretch;
}

.main-stack,
.side-stack {
  gap: 16px;
}

.chat-card {
  padding: 0 !important;
  overflow: hidden;
}

.composer-card {
  padding: 14px !important;
}

.composer-toolbar {
  margin-bottom: 10px;
  color: var(--ink-2);
  font-size: 13px;
}

.composer-input textarea {
  min-height: 76px !important;
  padding: 18px 18px 14px !important;
  border: 1px solid rgba(125, 98, 70, 0.16) !important;
  border-radius: 18px !important;
  background: rgba(255, 255, 255, 0.84) !important;
  color: var(--ink-1) !important;
  font-size: 15px !important;
  line-height: 1.68 !important;
  box-shadow: none !important;
}

.composer-input textarea::placeholder {
  color: #000000 !important;
  opacity: 0.72 !important;
}

.send-btn button,
.ghost-btn button {
  min-height: 52px;
  border-radius: 16px !important;
  font-weight: 700 !important;
}

.send-btn button {
  background: linear-gradient(135deg, #efc5a9, #e6b898) !important;
  border: none !important;
  color: #000000 !important;
}

.ghost-btn button {
  background: rgba(255, 255, 255, 0.68) !important;
  border: 1px solid rgba(125, 98, 70, 0.18) !important;
  color: #000000 !important;
}

.chat-shell {
  height: 620px;
  padding: 22px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.28), rgba(255, 255, 255, 0.06)),
    linear-gradient(180deg, #fffdf9 0%, #fbf5eb 100%);
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 0, 0, 0.25) transparent;
}

.chat-shell::-webkit-scrollbar {
  width: 10px;
}

.chat-shell::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.22);
  border-radius: 999px;
}

.chat-shell::-webkit-scrollbar-track {
  background: transparent;
}

.chat-empty {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  min-height: 100%;
  padding: 28px;
  border: 1px dashed rgba(125, 98, 70, 0.22);
  border-radius: 24px;
  background:
    radial-gradient(circle at top right, rgba(178, 90, 44, 0.08), transparent 28%),
    rgba(255, 255, 255, 0.64);
}

.chat-empty-kicker {
  display: inline-flex;
  padding: 6px 12px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}

.chat-empty h2 {
  margin: 14px 0 10px;
  color: var(--ink-1);
  font-size: 30px;
  line-height: 1.2;
  font-family: "Source Han Serif SC", "Noto Serif SC", "Songti SC", Georgia, serif;
}

.chat-empty p {
  margin: 0;
  max-width: 620px;
  color: var(--ink-2);
  font-size: 15px;
  line-height: 1.76;
}

.chat-thread {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.chat-row {
  display: flex;
}

.chat-row.user {
  justify-content: flex-end;
}

.chat-row.assistant {
  justify-content: flex-start;
}

.chat-bubble {
  max-width: min(86%, 780px);
  padding: 14px 16px 13px;
  border-radius: 22px;
  border: 1px solid rgba(125, 98, 70, 0.14);
  box-shadow: 0 14px 30px rgba(75, 55, 33, 0.08);
  color: #000000 !important;
}

.chat-bubble *,
.chat-meta,
.chat-content {
  color: #000000 !important;
}

.chat-row.user .chat-bubble {
  background: linear-gradient(135deg, #f4d9c4, #efcbb0);
  color: #000000;
  border-bottom-right-radius: 8px;
}

.chat-row.assistant .chat-bubble {
  background: rgba(255, 255, 255, 0.86);
  color: #000000;
  border-bottom-left-radius: 8px;
}

.chat-meta {
  margin-bottom: 8px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.chat-row.user .chat-meta {
  color: #000000;
}

.chat-row.assistant .chat-meta {
  color: #000000;
}

.chat-content {
  font-size: 15px;
  line-height: 1.8;
  white-space: normal;
  word-break: break-word;
}

.side-card {
  overflow: hidden;
}

.panel-shell {
  padding: 18px 18px 16px;
}

.panel-kicker {
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.panel-title {
  margin: 8px 0 0;
  color: var(--ink-1);
  font-size: 22px;
  line-height: 1.2;
  font-family: "Source Han Serif SC", "Noto Serif SC", "Songti SC", Georgia, serif;
}

.panel-desc {
  margin: 8px 0 16px;
  color: var(--ink-2);
  font-size: 14px;
  line-height: 1.7;
}

.status-list,
.memory-list,
.loop-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.status-item,
.memory-block,
.loop-item {
  padding: 14px;
  border-radius: 18px;
  border: 1px solid rgba(125, 98, 70, 0.14);
  background: rgba(255, 255, 255, 0.70);
}

.status-top,
.loop-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.status-name,
.loop-name {
  color: var(--ink-1);
  font-size: 14px;
  font-weight: 700;
}

.status-pill,
.loop-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 5px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.status-pill.ok,
.loop-pill.done {
  background: var(--olive-soft);
  color: var(--olive);
}

.status-pill.error {
  background: rgba(188, 58, 58, 0.12);
  color: #000000;
}

.loop-pill.step {
  background: var(--accent-soft);
  color: var(--accent);
}

.status-detail,
.memory-entry,
.loop-detail,
.empty-note {
  margin-top: 10px;
  color: var(--ink-2);
  font-size: 13px;
  line-height: 1.7;
  word-break: break-word;
}

.memory-section-label {
  display: inline-flex;
  margin-bottom: 10px;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(47, 93, 81, 0.10);
  color: var(--olive);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.memory-entry + .memory-entry {
  margin-top: 8px;
}

.thinking-card textarea {
  border-radius: 18px !important;
  background: rgba(255, 255, 255, 0.74) !important;
  color: #000000 !important;
}

@media (max-width: 1024px) {
  .hero-inner {
    flex-direction: column;
  }

  .hero-title {
    font-size: 36px;
  }

  .chat-shell {
    height: 520px;
  }
}

@media (max-width: 768px) {
  .gradio-container {
    padding: 14px 10px 22px !important;
  }

  .hero-inner,
  .panel-shell,
  .chat-shell {
    padding: 16px;
  }

  .hero-title {
    font-size: 30px;
  }

  .chat-empty {
    min-height: 100%;
    padding: 18px;
  }

  .chat-bubble {
    max-width: 100%;
  }

  .chat-shell {
    height: 460px;
  }
}
"""


def _import_gradio() -> Any:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError("gradio is required to launch the UI") from exc
    return gr


class NovelAgentApplication:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.workspace = WorkspaceManager(self.config)
        self.workspace.bootstrap()
        self.session_store = SessionStore(self.config.session_root, summary_max_chars=self.config.session_summary_max_chars)
        self.session_store.bootstrap()
        self.meta_store = SessionMetaStore(self.config.session_root)
        self.meta_store.bootstrap()
        self.registry = build_default_registry()
        self.decision_backend = LocalDecisionBackend(self.config)
        self.compression_backend = LocalCompressionBackend(self.config)
        self.controller = NovelAgentController(
            ControllerDependencies(
                config=self.config,
                workspace=self.workspace,
                registry=self.registry,
                decision_backend=self.decision_backend,
                compression_backend=self.compression_backend,
            )
        )
        self.heartbeat = HeartbeatManager(
            config=self.config,
            session_store=self.session_store,
            meta_store=self.meta_store,
            workspace=self.workspace,
        )
        self.heartbeat.run_idle_heartbeat_once(max_sessions=20)

    def new_session(self) -> SessionState:
        session = self.session_store.create_session()
        self.meta_store.get_or_create(session.session_id)
        return session

    def latest_or_new_session(self) -> SessionState:
        session = self.session_store.load_latest_session()
        if session is None:
            return self.new_session()
        meta = self.meta_store.load(session.session_id)
        if meta is not None and meta.cached_summary.strip():
            session.summary = meta.cached_summary.strip()
        return session

    def backend_status(self) -> dict[str, dict[str, Any]]:
        decision = self.decision_backend.healthcheck().model_dump()
        compression = self.compression_backend.healthcheck().model_dump()
        return {"decision_backend": decision, "compression_backend": compression}

    def handle_chat(self, session: SessionState, user_text: str) -> dict[str, Any]:
        result = self.controller.handle_user_message(session, user_text)
        self.session_store.append_events(session, result.transcript_events)
        self.heartbeat.run_turn_heartbeat(session)
        self.heartbeat.maybe_run_idle_heartbeat(max_sessions=5)
        return {
            "reply": result.reply,
            "transcript_events": result.transcript_events,
            "thinking": result.thinking if self.config.show_debug_thinking else "",
            "memory_preview": result.memory_preview,
            "session": session,
        }


def _append_chat_history(history: list[dict[str, str]], user_text: str, assistant_text: str):
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    return history


def _empty_history():
    return []


def _render_panel(*, kicker: str, title: str, description: str, body: str) -> str:
    return f"""
    <section class="panel-shell">
      <div class="panel-kicker">{html.escape(kicker)}</div>
      <h3 class="panel-title">{html.escape(title)}</h3>
      <p class="panel-desc">{html.escape(description)}</p>
      {body}
    </section>
    """


def _render_hero(app_title: str) -> str:
    return f"""
    <section class="hero-inner">
      <div class="hero-copy">
        <span class="hero-kicker">Closed-Domain Novel Agent</span>
        <h1 class="hero-title">{html.escape(app_title)}</h1>
        <p class="hero-subtitle">
          围绕小说理解、章节压缩与记忆管理构建的封闭式 agent。
          左侧专注对话，右侧追踪 loop 流程与系统状态，让调试与使用都更顺手。
        </p>
      </div>
      <div class="hero-tags">
        <span class="hero-tag">小说域对话</span>
        <span class="hero-tag">章节压缩 Tool</span>
        <span class="hero-tag">Loop 过程可视化</span>
        <span class="hero-tag">Markdown Memory</span>
      </div>
    </section>
    """


def _render_backend_status(status: dict[str, dict[str, Any]]) -> str:
    cards: list[str] = []
    for name, payload in status.items():
        state = "ok" if payload.get("ok") else "error"
        label = "OK" if payload.get("ok") else "ERROR"
        detail = html.escape(str(payload.get("detail", "") or ""))
        cards.append(
            f"""
            <div class="status-item">
              <div class="status-top">
                <div class="status-name">{html.escape(name)}</div>
                <span class="status-pill {state}">{label}</span>
              </div>
              <div class="status-detail">{detail}</div>
            </div>
            """
        )
    return _render_panel(
        kicker="runtime",
        title="运行状态",
        description="当前决策模型、压缩工具和本地后端的可用状态。",
        body=f'<div class="status-list">{"".join(cards)}</div>',
    )


def _render_loop_item(step_label: str, name: str, detail: str, pill_class: str = "step") -> str:
    return f"""
    <div class="loop-item">
      <div class="loop-top">
        <div class="loop-name">{html.escape(name)}</div>
        <span class="loop-pill {html.escape(pill_class)}">{html.escape(step_label)}</span>
      </div>
      <div class="loop-detail">{html.escape(detail)}</div>
    </div>
    """


def _render_loop_trace(events: list[dict[str, Any]]) -> str:
    items: list[str] = []
    for event in events:
        event_type = event.get("event_type")
        step_index = event.get("step_index", "?")

        if event_type == "agent_decision":
            payload = event.get("payload") or {}
            action = payload.get("action", "")
            tool_name = payload.get("tool_name") or ""
            if action == "call_tool":
                detail = f"决策调用工具 {tool_name}"
            elif action == "direct_reply":
                detail = "决策直接回复用户"
            elif action == "reject":
                detail = "决策拒绝当前请求"
            else:
                detail = f"决策执行 {action or 'unknown'}"
            items.append(_render_loop_item(f"Step {step_index}", "决策", detail))
            continue

        if event_type == "tool_call":
            tool_name = event.get("tool_name", "")
            items.append(_render_loop_item(f"Step {step_index}", "Tool Call", f"调用 {tool_name}"))
            continue

        if event_type == "tool_result":
            tool_name = event.get("tool_name", "")
            items.append(_render_loop_item(f"Step {step_index}", "Observation", f"接收 {tool_name} 返回结果"))
            continue

        if event_type == "assistant_message":
            items.append(_render_loop_item("Final", "输出", "将最终回复返回到前端", pill_class="done"))

    if not items:
        items.append('<div class="empty-note">等待新的请求。模型完成一次完整 loop 后，这里会显示本轮的决策与工具调用过程。</div>')

    return _render_panel(
        kicker="loop",
        title="模型 Loop 流程",
        description="只展示这一轮里模型做了什么，不展开正文结果。",
        body=f'<div class="loop-list">{"".join(items)}</div>',
    )


def _render_memory_preview(memory_preview: dict[str, list[str]]) -> str:
    daily_entries = memory_preview.get("daily") or []
    long_term_entries = memory_preview.get("long_term") or []
    blocks: list[str] = []

    if long_term_entries:
        blocks.append(
            """
            <div class="memory-block">
              <span class="memory-section-label">Long-Term</span>
            """
            + "".join(f'<div class="memory-entry">{html.escape(item)}</div>' for item in long_term_entries)
            + "</div>"
        )

    if daily_entries:
        blocks.append(
            """
            <div class="memory-block">
              <span class="memory-section-label">Daily</span>
            """
            + "".join(f'<div class="memory-entry">{html.escape(item)}</div>' for item in daily_entries)
            + "</div>"
        )

    if not blocks:
        blocks.append('<div class="empty-note">本轮没有新增需要写入 memory 的内容。</div>')

    return _render_panel(
        kicker="memory",
        title="本轮记忆写入",
        description="这里显示本次对话结束后追加到 daily memory 或 long-term memory 的摘要。",
        body=f'<div class="memory-list">{"".join(blocks)}</div>',
    )


def _render_chat_html(history: list[dict[str, str]]) -> str:
    if not history:
        return """
        <div class="chat-shell">
          <div class="chat-empty">
            <span class="chat-empty-kicker">Ready</span>
            <h2>从一段小说对话开始</h2>
            <p>你可以讨论剧情、人物、设定，也可以直接粘贴一章正文，让 agent 调用章节压缩工具。</p>
          </div>
        </div>
        """

    blocks = ['<div class="chat-shell"><div class="chat-thread">']
    for item in history:
        role = item.get("role", "assistant")
        content = html.escape(str(item.get("content", ""))).replace("\n", "<br>")
        if role == "user":
            blocks.append(
                f"""
                <div class="chat-row user">
                  <div class="chat-bubble">
                    <div class="chat-meta">你</div>
                    <div class="chat-content">{content}</div>
                  </div>
                </div>
                """
            )
        else:
            blocks.append(
                f"""
                <div class="chat-row assistant">
                  <div class="chat-bubble">
                    <div class="chat-meta">Novel Agent</div>
                    <div class="chat-content">{content}</div>
                  </div>
                </div>
                """
            )
    blocks.append("</div></div>")
    return "".join(blocks)


def build_demo(config: AgentConfig | None = None):
    gr = _import_gradio()
    app = NovelAgentApplication(config=config)
    initial_session = app.latest_or_new_session()
    initial_history = initial_session.chat_history()
    initial_status = _render_backend_status(app.backend_status())
    initial_loop = _render_loop_trace([])
    initial_memory = _render_memory_preview({"daily": [], "long_term": []})

    def submit_message(
        user_message: str,
        history: list[dict[str, str]] | None,
        session: SessionState | None,
    ):
        history = list(history or [])
        session = session or app.new_session()
        if not user_message or not user_message.strip():
            return (
                gr.skip(),
                gr.skip(),
                session,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

        outcome = app.handle_chat(session, user_message)
        history = _append_chat_history(history, user_message, outcome["reply"])

        return (
            _render_chat_html(history),
            history,
            outcome["session"],
            _render_loop_trace(outcome["transcript_events"]),
            _render_memory_preview(outcome["memory_preview"]),
            outcome["thinking"],
            _render_backend_status(app.backend_status()),
        )

    def reset_session():
        session = app.new_session()
        history = _empty_history()
        return (
            _render_chat_html(history),
            history,
            session,
            _render_loop_trace([]),
            _render_memory_preview({"daily": [], "long_term": []}),
            "",
            _render_backend_status(app.backend_status()),
        )

    with gr.Blocks(title=app.config.app_title, css=APP_CSS) as demo:
        gr.HTML(_render_hero(app.config.app_title), elem_classes="hero-card")
        with gr.Row(elem_classes="layout-row"):
            with gr.Column(scale=7, elem_classes="main-stack"):
                chatbot = gr.HTML(value=_render_chat_html(initial_history), elem_classes="chat-card")
                with gr.Column(elem_classes="composer-card"):
                    gr.HTML(
                        """
                        <div class="composer-toolbar">
                          输入
                        </div>
                        """
                    )
                    user_input = gr.Textbox(
                        placeholder="输入小说问题，或直接粘贴章节请求压缩",
                        lines=1,
                        max_lines=12,
                        show_label=False,
                        elem_classes="composer-input",
                    )
                    with gr.Row():
                        send_btn = gr.Button("发送", variant="primary", elem_classes="send-btn")
                        reset_btn = gr.Button("新建会话", elem_classes="ghost-btn")

            with gr.Column(scale=5, elem_classes="side-stack"):
                backend_status = gr.HTML(value=initial_status, elem_classes="side-card")
                loop_trace_box = gr.HTML(value=initial_loop, elem_classes="side-card")
                memory_box = gr.HTML(value=initial_memory, elem_classes="side-card")
                thinking_box = gr.Textbox(
                    label="Thinking（调试）",
                    visible=app.config.show_debug_thinking,
                    lines=8,
                    interactive=False,
                    elem_classes="thinking-card",
                )

        session_state = gr.State(initial_session)
        history_state = gr.State(initial_history)

        submit_event = gr.on(
            triggers=[send_btn.click, user_input.submit],
            fn=submit_message,
            inputs=[user_input, history_state, session_state],
            outputs=[chatbot, history_state, session_state, loop_trace_box, memory_box, thinking_box, backend_status],
            show_progress="minimal",
        )
        submit_event.then(lambda: "", outputs=user_input)

        reset_btn.click(
            fn=reset_session,
            outputs=[chatbot, history_state, session_state, loop_trace_box, memory_box, thinking_box, backend_status],
        )

    return demo


def main() -> None:
    try:
        demo = build_demo()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    demo.launch(server_name="0.0.0.0", server_port=7860)
