from __future__ import annotations

import html
import json
from typing import Any

from .backends import LocalCompressionBackend, LocalDecisionBackend
from .config import AgentConfig
from .controller import ControllerDependencies, NovelAgentController
from .memory import SessionState
from .registry import build_default_registry
from .workspace import WorkspaceManager


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

    def new_session(self) -> SessionState:
        return SessionState()

    def backend_status(self) -> dict[str, dict[str, Any]]:
        decision = self.decision_backend.healthcheck().model_dump()
        compression = self.compression_backend.healthcheck().model_dump()
        return {"decision_backend": decision, "compression_backend": compression}

    def handle_chat(self, session: SessionState, user_text: str) -> dict[str, Any]:
        result = self.controller.handle_user_message(session, user_text)
        return {
            "reply": result.reply,
            "domain": result.domain,
            "action": result.action,
            "decision": result.decision,
            "tool_trace": result.tool_trace,
            "thinking": result.thinking if self.config.show_debug_thinking else "",
            "memory_preview": result.memory_preview,
            "session": session,
        }


def _render_backend_status(status: dict[str, dict[str, Any]]) -> str:
    lines = []
    for name, payload in status.items():
        state = "OK" if payload.get("ok") else "ERROR"
        lines.append(f"{name}: {state} | {payload.get('detail', '')}")
    return "\n".join(lines)


def _append_chat_history(history: list[dict[str, str]], user_text: str, assistant_text: str):
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    return history


def _empty_history():
    return []


def _render_chat_html(history: list[dict[str, str]]) -> str:
    blocks = [
        """
        <div style="display:flex; flex-direction:column; gap:12px; padding:12px; background:#111827; min-height:520px; border-radius:12px;">
        """
    ]
    for item in history:
        role = item.get("role", "assistant")
        content = html.escape(str(item.get("content", ""))).replace("\n", "<br>")
        if role == "user":
            blocks.append(
                f"""
                <div style="display:flex; justify-content:flex-end;">
                  <div style="max-width:85%; background:#2563eb; color:white; padding:10px 12px; border-radius:14px 14px 4px 14px;">
                    {content}
                  </div>
                </div>
                """
            )
        else:
            blocks.append(
                f"""
                <div style="display:flex; justify-content:flex-start;">
                  <div style="max-width:85%; background:#1f2937; color:#f9fafb; padding:10px 12px; border-radius:14px 14px 14px 4px; border:1px solid #374151;">
                    {content}
                  </div>
                </div>
                """
            )
    blocks.append("</div>")
    return "".join(blocks)


def build_demo(config: AgentConfig | None = None):
    gr = _import_gradio()
    app = NovelAgentApplication(config=config)

    def submit_message(
        user_message: str,
        history: list[dict[str, str]] | None,
        session: SessionState | None,
    ):
        history = list(history or [])
        session = session or app.new_session()
        if not user_message or not user_message.strip():
            return _render_chat_html(history), history, session, "", "", "", "", "", _render_backend_status(app.backend_status())

        outcome = app.handle_chat(session, user_message)
        history = _append_chat_history(history, user_message, outcome["reply"])

        return (
            _render_chat_html(history),
            history,
            outcome["session"],
            outcome["domain"],
            outcome["action"],
            json.dumps(outcome["decision"], ensure_ascii=False, indent=2),
            json.dumps(outcome["tool_trace"], ensure_ascii=False, indent=2),
            outcome["thinking"],
            _render_backend_status(app.backend_status()),
        )

    def reset_session():
        session = app.new_session()
        history = _empty_history()
        return _render_chat_html(history), history, session, "", "", "", "", "", _render_backend_status(app.backend_status())

    with gr.Blocks(title=app.config.app_title) as demo:
        gr.Markdown("# Novel Agent V1")
        gr.Markdown("仅服务小说域。小说域外统一返回：`无法回答`。")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.HTML(value=_render_chat_html([]), label="Novel Agent")
                user_input = gr.Textbox(label="输入", placeholder="输入小说问题，或直接粘贴章节请求压缩", lines=4)
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    reset_btn = gr.Button("重置会话")
            with gr.Column(scale=2):
                backend_status = gr.Textbox(label="Backend 状态", value=_render_backend_status(app.backend_status()), lines=6)
                domain_box = gr.Textbox(label="当前 domain", interactive=False)
                action_box = gr.Textbox(label="当前 action", interactive=False)
                tool_trace = gr.Textbox(label="Tool Trace", lines=10, interactive=False)
                decision_json = gr.Textbox(label="Decision JSON", lines=12, interactive=False)
                thinking_box = gr.Textbox(
                    label="Thinking（调试）",
                    visible=app.config.show_debug_thinking,
                    lines=8,
                    interactive=False,
                )

        session_state = gr.State(app.new_session())
        history_state = gr.State(_empty_history())

        send_btn.click(
            fn=submit_message,
            inputs=[user_input, history_state, session_state],
            outputs=[chatbot, history_state, session_state, domain_box, action_box, decision_json, tool_trace, thinking_box, backend_status],
        ).then(lambda: "", outputs=user_input)

        user_input.submit(
            fn=submit_message,
            inputs=[user_input, history_state, session_state],
            outputs=[chatbot, history_state, session_state, domain_box, action_box, decision_json, tool_trace, thinking_box, backend_status],
        ).then(lambda: "", outputs=user_input)

        reset_btn.click(
            fn=reset_session,
            outputs=[chatbot, history_state, session_state, domain_box, action_box, decision_json, tool_trace, thinking_box, backend_status],
        )

    return demo


def main() -> None:
    try:
        demo = build_demo()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    demo.launch(server_name="0.0.0.0", server_port=7860)
