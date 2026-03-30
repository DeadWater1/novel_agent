from __future__ import annotations

from pydantic import BaseModel

from novel_agent.prompts import build_decision_system_prompt, build_plan_system_prompt
from novel_agent.registry import build_default_registry
from novel_agent.toolbox import ToolExecutionResult, ToolHandler, ToolRuntimeContext
from novel_agent.memory import SessionState


class FakeToolArgs(BaseModel):
    topic: str


class FakeToolHandler(ToolHandler):
    name = "fake_tool"
    description = "Fake tool for dynamic registry prompt tests."
    args_model = FakeToolArgs
    prompt_doc = """
## fake_tool

- 名称：`fake_tool`
- 作用：测试 registry 是否能动态生成 prompt 文档
- 输入：
  - `topic`: 必填
""".strip()

    def execute(
        self,
        session: SessionState,
        parsed_args: FakeToolArgs,
        runtime: ToolRuntimeContext,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=self.name,
            observation_text=parsed_args.topic,
            payload={"topic": parsed_args.topic},
            tool_trace={"requested_tool": self.name, "status": "ok"},
        )


def test_registry_renders_dynamic_prompt_docs_and_names():
    registry = build_default_registry()
    registry.register(FakeToolHandler())

    prompt_docs = registry.render_prompt_docs()
    tool_names = registry.names()

    assert "fake_tool" in tool_names
    assert "## fake_tool" in prompt_docs
    assert "测试 registry 是否能动态生成 prompt 文档" in prompt_docs


def test_prompt_builders_use_registry_derived_tool_docs():
    registry = build_default_registry()
    registry.register(FakeToolHandler())

    plan_prompt = build_plan_system_prompt("WORKSPACE", registry.render_prompt_docs(), registry.names())
    decision_prompt = build_decision_system_prompt("WORKSPACE", registry.render_prompt_docs(), registry.names())

    assert "fake_tool" in plan_prompt
    assert "fake_tool" in decision_prompt
    assert "当前可用工具只有" in plan_prompt
    assert "当前可用工具只有" in decision_prompt
