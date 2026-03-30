from __future__ import annotations

import html
import os
import threading
import time
from typing import Any

from .backends import LocalEmbeddingBackend, build_generation_backends
from .compaction import ContextCompactionManager
from .config import AgentConfig
from .controller import ControllerDependencies, NovelAgentController
from .embedding_index import EmbeddingIndexManager
from .heartbeat import HeartbeatManager
from .memory import SessionState, SessionStore
from .registry import build_default_registry
from .session_meta import SessionMetaStore
from .workspace import WorkspaceManager

DEFAULT_SERVER_PORT_CANDIDATES = (7860, 7861, 7862, 9888, 9988, 17860, 27860)


APP_CSS = """
:root {
  --paper-1: #050505;
  --paper-2: #0b0908;
  --paper-3: rgba(20, 16, 13, 0.92);
  --ink-1: #ffffff;
  --ink-2: rgba(255, 255, 255, 0.82);
  --line: rgba(255, 255, 255, 0.08);
  --accent: #ffffff;
  --accent-soft: rgba(255, 255, 255, 0.08);
  --olive: #ffffff;
  --olive-soft: rgba(255, 255, 255, 0.08);
  --shadow: 0 24px 60px rgba(0, 0, 0, 0.42);
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(180, 128, 52, 0.10), transparent 30%),
    radial-gradient(circle at top center, rgba(117, 70, 36, 0.10), transparent 24%),
    radial-gradient(circle at right center, rgba(255, 255, 255, 0.03), transparent 28%),
    linear-gradient(180deg, #050505 0%, #080706 52%, #040404 100%);
  color: var(--ink-1);
  font-family: "IBM Plex Sans", "Noto Sans SC", "PingFang SC", sans-serif;
}

.gradio-container {
  max-width: 1560px !important;
  padding: 24px 20px 32px !important;
}

.gradio-container * {
  color: var(--ink-1);
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
  backdrop-filter: blur(14px);
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
    linear-gradient(135deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01)),
    radial-gradient(circle at top left, rgba(175, 125, 56, 0.12), transparent 34%),
    linear-gradient(120deg, rgba(65, 46, 30, 0.32), rgba(12, 10, 9, 0.18));
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
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.06);
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
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 18px !important;
  background: rgba(255, 255, 255, 0.04) !important;
  color: var(--ink-1) !important;
  font-size: 15px !important;
  line-height: 1.68 !important;
  box-shadow: none !important;
}

.composer-input textarea::placeholder {
  color: rgba(255, 255, 255, 0.48) !important;
  opacity: 1 !important;
}

.send-btn button,
.ghost-btn button {
  min-height: 52px;
  border-radius: 16px !important;
  font-weight: 700 !important;
}

.send-btn button {
  background: linear-gradient(135deg, #2b231b, #181311) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  color: #ffffff !important;
}

.ghost-btn button {
  background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  color: #ffffff !important;
}

.chat-shell {
  height: 760px;
  padding: 22px;
  background:
    radial-gradient(circle at top center, rgba(186, 128, 60, 0.06), transparent 24%),
    linear-gradient(180deg, rgba(17, 14, 12, 0.94), rgba(8, 7, 7, 0.98)),
    linear-gradient(180deg, #0b0a09 0%, #050505 100%);
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-width: thin;
  scrollbar-color: rgba(255, 255, 255, 0.18) transparent;
}

.chat-shell::-webkit-scrollbar {
  width: 10px;
}

.chat-shell::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.18);
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
  border: 1px dashed rgba(255, 255, 255, 0.12);
  border-radius: 24px;
  background:
    radial-gradient(circle at top right, rgba(178, 109, 54, 0.10), transparent 28%),
    rgba(255, 255, 255, 0.02);
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
  max-width: min(92%, 980px);
  padding: 14px 16px 13px;
  border-radius: 22px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 14px 30px rgba(0, 0, 0, 0.26);
  color: #ffffff !important;
}

.chat-bubble *,
.chat-meta,
.chat-content {
  color: #ffffff !important;
}

.chat-row.user .chat-bubble {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.05));
  color: #ffffff;
  border-bottom-right-radius: 8px;
}

.chat-row.assistant .chat-bubble {
  background: rgba(255, 255, 255, 0.04);
  color: #ffffff;
  border-bottom-left-radius: 8px;
}

.chat-meta {
  margin-bottom: 8px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.chat-row.user .chat-meta {
  color: rgba(255, 255, 255, 0.72);
}

.chat-row.assistant .chat-meta {
  color: rgba(255, 255, 255, 0.72);
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

.panel-body {
  min-width: 0;
}

.panel-body-scroll {
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-width: thin;
  scrollbar-color: rgba(255, 255, 255, 0.18) transparent;
}

.panel-body-scroll::-webkit-scrollbar {
  width: 8px;
}

.panel-body-scroll::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.18);
  border-radius: 999px;
}

.panel-body-scroll::-webkit-scrollbar-track {
  background: transparent;
}

.runtime-scroll {
  max-height: 220px;
  padding-right: 4px;
}

.loop-scroll {
  max-height: 320px;
  padding-right: 4px;
}

.decision-scroll {
  max-height: 260px;
  padding-right: 4px;
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
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.04);
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
  color: #ffffff;
}

.status-pill.error {
  background: rgba(188, 58, 58, 0.22);
  color: #ffffff;
}

.loop-pill.step {
  background: var(--accent-soft);
  color: #ffffff;
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
  background: rgba(255, 255, 255, 0.08);
  color: #ffffff;
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
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  background: rgba(255, 255, 255, 0.04) !important;
  color: #ffffff !important;
}

button:hover,
.chat-bubble:hover,
.status-item:hover,
.memory-block:hover,
.loop-item:hover {
  filter: brightness(1.06);
}

@media (max-width: 1024px) {
  .hero-inner {
    flex-direction: column;
  }

  .hero-title {
    font-size: 36px;
  }

  .chat-shell {
    height: 620px;
  }

  .runtime-scroll {
    max-height: 200px;
  }

  .loop-scroll {
    max-height: 280px;
  }

  .decision-scroll {
    max-height: 240px;
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
    height: 540px;
  }

  .runtime-scroll {
    max-height: 180px;
  }

  .loop-scroll {
    max-height: 260px;
  }

  .decision-scroll {
    max-height: 220px;
  }
}
"""


APP_HEAD = """
<script>
(() => {
  const CHAT_ROOT_ID = "chatbot-view";

  function scrollChatToBottom(container) {
    const shell = container?.querySelector(".chat-shell");
    if (!shell) {
      return;
    }
    shell.scrollTop = shell.scrollHeight;
  }

  function bindChatAutoScroll() {
    const container = document.getElementById(CHAT_ROOT_ID);
    if (!container) {
      return false;
    }
    scrollChatToBottom(container);
    if (container.dataset.autoscrollBound === "1") {
      return true;
    }
    container.dataset.autoscrollBound = "1";
    const observer = new MutationObserver(() => {
      window.requestAnimationFrame(() => scrollChatToBottom(container));
    });
    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    return true;
  }

  function installAutoScroll() {
    if (bindChatAutoScroll()) {
      return;
    }
    const pageObserver = new MutationObserver(() => {
      if (bindChatAutoScroll()) {
        pageObserver.disconnect();
      }
    });
    pageObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installAutoScroll, { once: true });
  } else {
    installAutoScroll();
  }
})();
</script>
"""


CAPTURE_AND_CLEAR_INPUT_JS = """
(user_message) => {
  const clean = typeof user_message === "string" ? user_message : "";
  return [clean, ""];
}
""".strip()

PROCESSING_PLACEHOLDER = "正在处理..."


def _import_gradio() -> Any:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError("gradio is required to launch the UI") from exc
    return gr


class NovelAgentApplication:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()
        self.embedding_backend = LocalEmbeddingBackend(self.config)
        self.embedding_index_manager = EmbeddingIndexManager(self.config, self.embedding_backend)
        self.embedding_index_manager.bootstrap()
        self.workspace = WorkspaceManager(
            self.config,
            embedding_backend=self.embedding_backend,
            embedding_index_manager=self.embedding_index_manager,
        )
        self.workspace.bootstrap()
        self.session_store = SessionStore(self.config.session_root, summary_max_chars=self.config.session_summary_max_chars)
        self.session_store.bootstrap()
        self.meta_store = SessionMetaStore(self.config.session_root)
        self.meta_store.bootstrap()
        self.registry = build_default_registry(self.config.tool_registry_enabled)
        self.decision_backend, self.compression_backend, self.summary_backend = build_generation_backends(self.config)
        self.compaction_manager = ContextCompactionManager(
            self.config,
            self.session_store,
            self.meta_store,
            summary_backend=self.summary_backend,
        )
        self.compaction_manager.bootstrap()
        self.controller = NovelAgentController(
            ControllerDependencies(
                config=self.config,
                workspace=self.workspace,
                registry=self.registry,
                decision_backend=self.decision_backend,
                compression_backend=self.compression_backend,
                embedding_backend=self.embedding_backend,
                embedding_index_manager=self.embedding_index_manager,
                session_store=self.session_store,
                meta_store=self.meta_store,
                compaction_manager=self.compaction_manager,
            )
        )
        self.heartbeat = HeartbeatManager(
            config=self.config,
            session_store=self.session_store,
            meta_store=self.meta_store,
            workspace=self.workspace,
            compaction_manager=self.compaction_manager,
            embedding_index_manager=self.embedding_index_manager,
        )
        self._startup_maintenance_started = False

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
        embedding = self.embedding_backend.healthcheck().model_dump()
        compression = self.compression_backend.healthcheck().model_dump()
        summary = self.summary_backend.healthcheck().model_dump()
        return {
            "decision_backend": decision,
            "embedding_backend": embedding,
            "compression_backend": compression,
            "summary_backend": summary,
        }

    def handle_chat(self, session: SessionState, user_text: str) -> dict[str, Any]:
        result = self.controller.handle_user_message(session, user_text)
        self.session_store.append_events(session, result.transcript_events)
        self.heartbeat.run_turn_heartbeat(session)
        self.heartbeat.maybe_run_idle_heartbeat(max_sessions=5)
        return {
            "reply": result.reply,
            "transcript_events": result.transcript_events,
            "thinking": result.thinking if self.config.show_debug_thinking else "",
            "plan_output_text": str(getattr(self.decision_backend, "last_plan_output_text", "") or ""),
            "decision_output_text": str(getattr(self.decision_backend, "last_decision_output_text", "") or ""),
            "review_output_text": str(getattr(self.decision_backend, "last_review_output_text", "") or ""),
            "memory_preview": result.memory_preview,
            "context_report": result.context_report,
            "session": session,
        }

    def start_startup_maintenance(self, *, delay_seconds: float = 3.0, max_sessions: int = 20) -> None:
        if self._startup_maintenance_started:
            return
        self._startup_maintenance_started = True

        def _worker() -> None:
            time.sleep(delay_seconds)
            try:
                self.heartbeat.run_idle_heartbeat_once(max_sessions=max_sessions)
            except Exception:
                # Startup maintenance is best-effort and should never block the UI.
                return

        threading.Thread(target=_worker, name="novel-agent-startup-maintenance", daemon=True).start()


def _append_chat_history(history: list[dict[str, str]], user_text: str, assistant_text: str):
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    return history


def _append_pending_chat_history(history: list[dict[str, str]], user_text: str):
    return _append_chat_history(history, user_text, PROCESSING_PLACEHOLDER)


def _finalize_chat_history(history: list[dict[str, str]], user_text: str, assistant_text: str):
    if (
        len(history) >= 2
        and history[-2].get("role") == "user"
        and history[-2].get("content") == user_text
        and history[-1].get("role") == "assistant"
        and history[-1].get("content") == PROCESSING_PLACEHOLDER
    ):
        history[-1] = {"role": "assistant", "content": assistant_text}
        return history
    return _append_chat_history(history, user_text, assistant_text)


def _empty_history():
    return []


def _render_panel(
    *,
    kicker: str,
    title: str,
    description: str,
    body: str,
    body_class: str = "",
) -> str:
    classes = "panel-body"
    if body_class.strip():
        classes = f"{classes} {body_class.strip()}"
    return f"""
    <section class="panel-shell">
      <div class="panel-kicker">{html.escape(kicker)}</div>
      <h3 class="panel-title">{html.escape(title)}</h3>
      <p class="panel-desc">{html.escape(description)}</p>
      <div class="{html.escape(classes)}">{body}</div>
    </section>
    """


def _render_hero(app_title: str) -> str:
    return f"""
    <section class="hero-inner">
      <div class="hero-copy">
        <h1 class="hero-title">{html.escape(app_title)}</h1>
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
        body_class="panel-body-scroll runtime-scroll",
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

        if event_type == "plan_created":
            items.append(_render_loop_item("Plan", "Plan Created", "生成本轮执行计划"))
            continue

        if event_type == "plan_updated":
            items.append(_render_loop_item(f"Step {step_index}", "Plan Updated", "执行中更新剩余计划"))
            continue

        if event_type == "plan_update_ignored":
            items.append(_render_loop_item(f"Step {step_index}", "Plan Update Ignored", "本轮重规划次数已用尽"))
            continue

        if event_type == "plan_step_completed":
            payload = event.get("payload") or {}
            goal = payload.get("goal", "") or event.get("goal", "")
            detail = f"完成 step：{goal}" if goal else "完成当前 step"
            items.append(_render_loop_item(f"Step {step_index}", "Step Completed", detail))
            continue

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

        if event_type == "decision_review":
            payload = event.get("payload") or {}
            verdict = payload.get("verdict", "accept")
            detail = "首答通过复审" if verdict == "accept" else "首答被要求重试"
            items.append(_render_loop_item(f"Step {step_index}", "Review", detail))
            continue

        if event_type == "premature_direct_reply_blocked":
            items.append(_render_loop_item(f"Step {step_index}", "Direct Reply Blocked", "当前计划尚未完成，阻止提前结束"))
            continue

        if event_type == "terminal_tool_deferred":
            tool_name = event.get("tool_name", "")
            detail = f"{tool_name} 已返回结果，但当前仅作为 observation 继续后续步骤" if tool_name else "终局工具结果被降级为 observation"
            items.append(_render_loop_item(f"Step {step_index}", "Terminal Tool Deferred", detail))
            continue

        if event_type == "final_synthesis_started":
            items.append(_render_loop_item(f"Step {step_index}", "Final Synthesis", "基于已有工具证据生成最终答复"))
            continue

        if event_type == "final_synthesis_evidence":
            items.append(_render_loop_item(f"Step {step_index}", "Synthesis Evidence", "向决策模型回喂结构化工具证据"))
            continue

        if event_type == "final_synthesis_retry":
            items.append(_render_loop_item(f"Step {step_index}", "Synthesis Retry", "首次收口未成功，重试最终答复"))
            continue

        if event_type == "final_synthesis_failed":
            items.append(_render_loop_item(f"Step {step_index}", "Synthesis Failed", "工具证据已回喂，但最终答复仍未生成"))
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
        body_class="panel-body-scroll loop-scroll",
    )


def _render_decision_output(
    plan_output_text: str = "",
    decision_output_text: str = "",
    review_output_text: str = "",
) -> str:
    blocks: list[str] = []
    clean_plan = plan_output_text.strip()
    clean_decision = decision_output_text.strip()
    clean_review = review_output_text.strip()

    if clean_plan:
        blocks.append(
            """
            <div class="memory-block">
              <span class="memory-section-label">Planner Output</span>
            """
            + f'<div class="memory-entry">{html.escape(clean_plan).replace(chr(10), "<br>")}</div>'
            + "</div>"
        )

    if clean_decision:
        blocks.append(
            """
            <div class="memory-block">
              <span class="memory-section-label">Decision Output</span>
            """
            + f'<div class="memory-entry">{html.escape(clean_decision).replace(chr(10), "<br>")}</div>'
            + "</div>"
        )

    if clean_review:
        blocks.append(
            """
            <div class="memory-block">
              <span class="memory-section-label">Review Output</span>
            """
            + f'<div class="memory-entry">{html.escape(clean_review).replace(chr(10), "<br>")}</div>'
            + "</div>"
        )

    if not blocks:
        blocks.append('<div class="empty-note">当前没有可显示的决策模型原始输出。</div>')

    return _render_panel(
        kicker="decision",
        title="决策模型输出",
        description="展示 planner、决策模型与复审模型本轮的原始输出文本。",
        body=f'<div class="memory-list">{"".join(blocks)}</div>',
        body_class="panel-body-scroll decision-scroll",
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


def _render_context_report(report: dict[str, Any] | None) -> str:
    payload = dict(report or {})
    rows = [
        f'<div class="memory-entry">estimated_tokens: {html.escape(str(payload.get("estimated_tokens", 0)))}</div>',
        f'<div class="memory-entry">pruning_applied: {html.escape(str(bool(payload.get("pruning_applied", False))))}</div>',
        f'<div class="memory-entry">compaction_applied: {html.escape(str(bool(payload.get("compaction_applied", False))))}</div>',
        f'<div class="memory-entry">compaction_source: {html.escape(str(payload.get("compaction_source", "") or "(none)"))}</div>',
        f'<div class="memory-entry">memory_flush_applied: {html.escape(str(bool(payload.get("memory_flush_applied", False))))}</div>',
        f'<div class="memory-entry">review_triggered: {html.escape(str(bool(payload.get("review_triggered", False))))}</div>',
        f'<div class="memory-entry">review_verdict: {html.escape(str(payload.get("review_verdict", "") or "(none)"))}</div>',
    ]

    recall_targets = payload.get("recall_targets") or []
    if recall_targets:
        rows.extend(f'<div class="memory-entry">recall: {html.escape(str(item))}</div>' for item in recall_targets)

    context_blocks = payload.get("context_blocks") or []
    if context_blocks:
        rows.extend(f'<div class="memory-entry">block: {html.escape(str(item))}</div>' for item in context_blocks)

    if payload.get("memory_flush_daily"):
        rows.extend(
            f'<div class="memory-entry">flush_daily: {html.escape(str(item))}</div>'
            for item in payload.get("memory_flush_daily", [])
        )
    if payload.get("memory_flush_long_term"):
        rows.extend(
            f'<div class="memory-entry">flush_long_term: {html.escape(str(item))}</div>'
            for item in payload.get("memory_flush_long_term", [])
        )

    return _render_panel(
        kicker="context",
        title="Context Report",
        description="展示本轮上下文装配、自动召回、review 与 compact/flush 的摘要。",
        body=f'<div class="memory-list">{"".join(rows)}</div>',
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
    initial_decision_output = _render_decision_output()
    app.start_startup_maintenance()

    def prepare_submission(user_message: str):
        return (user_message or ""), ""

    def submit_message(
        user_message: str,
        history: list[dict[str, str]] | None,
        session: SessionState | None,
    ):
        history = list(history or [])
        session = session or app.new_session()
        if not user_message or not user_message.strip():
            yield (
                gr.skip(),
                gr.skip(),
                session,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
            )
            return

        history = _append_pending_chat_history(history, user_message)
        yield (
            _render_chat_html(history),
            history,
            session,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "",
        )

        outcome = app.handle_chat(session, user_message)
        history = _finalize_chat_history(history, user_message, outcome["reply"])

        yield (
            _render_chat_html(history),
            history,
            outcome["session"],
            _render_loop_trace(outcome["transcript_events"]),
            _render_decision_output(
                outcome["plan_output_text"],
                outcome["decision_output_text"],
                outcome["review_output_text"],
            ),
            _render_backend_status(app.backend_status()),
            "",
        )

    def reset_session():
        session = app.new_session()
        history = _empty_history()
        return (
            _render_chat_html(history),
            history,
            session,
            _render_loop_trace([]),
            _render_decision_output(),
            _render_backend_status(app.backend_status()),
            "",
        )

    with gr.Blocks(title=app.config.app_title) as demo:
        gr.HTML(_render_hero(app.config.app_title), elem_classes="hero-card")
        with gr.Row(elem_classes="layout-row"):
            with gr.Column(scale=8, elem_classes="main-stack"):
                chatbot = gr.HTML(
                    value=_render_chat_html(initial_history),
                    elem_classes="chat-card",
                    elem_id="chatbot-view",
                )
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

            with gr.Column(scale=4, elem_classes="side-stack"):
                backend_status = gr.HTML(value=initial_status, elem_classes="side-card")
                loop_trace_box = gr.HTML(value=initial_loop, elem_classes="side-card")
                decision_output_box = gr.HTML(value=initial_decision_output, elem_classes="side-card")

        session_state = gr.State(initial_session)
        history_state = gr.State(initial_history)
        pending_input_state = gr.State("")

        submit_event = gr.on(
            triggers=[send_btn.click, user_input.submit],
            fn=prepare_submission,
            js=CAPTURE_AND_CLEAR_INPUT_JS,
            inputs=[user_input],
            outputs=[pending_input_state, user_input],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=submit_message,
            inputs=[pending_input_state, history_state, session_state],
            outputs=[
                chatbot,
                history_state,
                session_state,
                loop_trace_box,
                decision_output_box,
                backend_status,
                pending_input_state,
            ],
            show_progress="minimal",
        )

        reset_btn.click(
            fn=reset_session,
            outputs=[
                chatbot,
                history_state,
                session_state,
                loop_trace_box,
                decision_output_box,
                backend_status,
                pending_input_state,
            ],
        )

    return demo


def main() -> None:
    try:
        demo = build_demo()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    for key in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in os.getenv(key, "").split(",") if item.strip()]
        for candidate in ("127.0.0.1", "localhost"):
            if candidate not in existing:
                existing.append(candidate)
        os.environ[key] = ",".join(existing)
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port_text = os.getenv("GRADIO_SERVER_PORT", "").strip()
    launch_kwargs: dict[str, Any] = {
        "server_name": server_name,
        "css": APP_CSS,
        "head": APP_HEAD,
    }
    if server_port_text:
        launch_kwargs["server_port"] = int(server_port_text)
        demo.launch(**launch_kwargs)
        return

    last_error: Exception | None = None
    for server_port in DEFAULT_SERVER_PORT_CANDIDATES:
        try:
            demo.launch(**launch_kwargs, server_port=server_port)
            return
        except OSError as exc:
            last_error = exc
            continue

    ports_text = ", ".join(str(port) for port in DEFAULT_SERVER_PORT_CANDIDATES)
    if last_error is not None:
        raise SystemExit(f"无法启动服务：候选端口均不可用（{ports_text}）。最后错误：{last_error}") from last_error
    raise SystemExit(f"无法启动服务：候选端口均不可用（{ports_text}）。")
