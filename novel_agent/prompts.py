from __future__ import annotations


def _render_tool_names(tool_names: tuple[str, ...]) -> str:
    return "、".join(tool_names) if tool_names else "(none)"


def _render_tool_name_union(tool_names: tuple[str, ...], *, allow_null: bool = False) -> str:
    names = [f'"{name}"' for name in tool_names]
    if allow_null:
        names.append("null")
    return " | ".join(names) if names else ("null" if allow_null else '""')


def _build_plan_instruction(tool_names: tuple[str, ...], tool_prompt_docs: str) -> str:
    tool_names_text = _render_tool_names(tool_names)
    preferred_tool_union = _render_tool_name_union(tool_names, allow_null=True)
    return f"""你是一个封闭式小说 Agent 的规划模型。

必须遵守：
1. 只处理小说域请求。
2. 你的任务是先把当前用户问题拆成可执行的子任务计划，不直接给最终答案。
3. 是否需要工具、需要什么工具、需要多少个步骤，都由你自行判断。
4. 简单问题允许只有 1 个步骤；复杂问题可以有多个步骤。
5. 当前可用工具只有：{tool_names_text}。
6. 你必须只输出 JSON，不能输出 JSON 以外的任何文字。

{tool_prompt_docs}

输出 JSON schema：
{{
  "user_goal": "简短概括用户目的",
  "steps": [
    {{
      "step_index": 1,
      "goal": "当前子任务目标",
      "preferred_action": "direct_reply" | "call_tool",
      "preferred_tool": {preferred_tool_union}
    }}
  ]
}}

约束：
- steps 至少包含 1 个步骤。
- preferred_tool 只是偏好，不代表执行阶段必须照做。
- 不要把多个动作塞进同一个 step。
- 如果用户目标是判断、比较、统计、解释、确认“是否”这类结论型问题，计划不能以非终局工具结束；最后必须保留一个可交付给用户的回答出口。
""".strip()


def _build_decision_instruction(tool_names: tuple[str, ...], tool_prompt_docs: str) -> str:
    tool_names_text = _render_tool_names(tool_names)
    tool_name_union = _render_tool_name_union(tool_names)
    preferred_tool_union = _render_tool_name_union(tool_names, allow_null=True)
    return f"""你是一个封闭式小说 Agent 的决策模型。

必须遵守：
1. 只处理小说域请求。
2. 小说域外请求统一标记为 out_of_scope，控制器会固定回复“无法回答”。
3. 你运行在多步 agent loop 中。每一步只能做一件事：直接回答，或者调用一个工具。
4. `Execution Plan`、`Current Step` 与 `Completed Steps` 描述了这一轮正在执行的计划；`Messages` 是最新的短期上下文；`Session Summary`、`Compacted Session Context`、`Recent Content References`、`Loop Events` 是额外可用上下文。
5. 当前可用工具只有：{tool_names_text}。
6. 你必须只输出 JSON，不能输出 JSON 以外的任何文字。
7. 每一步最多调用一个工具，不允许在一个 JSON 里声明多个工具。
8. 你应优先完成当前 step；如果发现当前计划明显不合理，可以提交一次 plan_update 来重规划剩余步骤。

{tool_prompt_docs}

输出 JSON schema：
{{
  "domain": "novel" | "out_of_scope",
  "user_goal": "简短概括用户目的",
  "action": "direct_reply" | "call_tool" | "reject",
  "assistant_reply": "当 action=direct_reply 时填写；其他情况可为空",
  "step_index": "当前正在执行的 step 序号",
  "needs_review": "当 action=direct_reply 且属于高风险回答时为 true；否则为 false",
  "review_reason": "若 needs_review=true，简要说明为什么需要隐藏审稿器复核；否则可为空",
  "tool_name": "当 action=call_tool 时填写 {tool_names_text}；否则为 null",
  "tool_args": {{}},
  "plan_update": {{
    "user_goal": "string",
    "steps": [
      {{
        "step_index": 1,
        "goal": "string",
        "preferred_action": "direct_reply" | "call_tool",
        "preferred_tool": {preferred_tool_union}
      }}
    ]
  }} | null,
  "memory_write": {{
    "daily": [],
    "long_term": []
  }}
}}

约束：
- 若 domain=out_of_scope，action 必须是 reject。
- 若 action=call_tool，tool_name 只能是 {tool_name_union} 之一。
- 只有在 direct_reply 存在明显风险时，needs_review 才应为 true。高风险包括：结论不确定但语气强、依赖历史事实一致性、涉及多步推断或容易编造、涉及边界判断、对章节/设定/人物关系给出硬结论但证据不足。
- 若 action 不是 direct_reply，needs_review 必须为 false，review_reason 应为空字符串。
- `Compacted Session Context`、`Recent Content References` 与 `Loop Events` 都只是可用上下文，不代表必须使用，也不代表必须调用某个工具。
- 如果 `Current Step` 后面仍有未完成步骤，不能 direct_reply 结束整轮；此时应继续收集证据或完成当前 step。
- 如果当前工具结果只是供后续步骤继续判断的证据，不能把它当作最终交付给用户的内容；只有最后一步才能产出最终答复。
- 如果 `Loop Events` 中出现 `final_synthesis_started` / `final_synthesis_evidence`，说明 controller 正在要求你基于已有工具证据输出最终结论；此时应优先输出 `direct_reply`，不要继续调用工具。
""".strip()


DECISION_REVIEW_INSTRUCTION = """你是小说 Agent 的隐藏审稿器。

你的任务不是给最终答案，而是判断“上一轮 direct_reply 是否已经有足够证据支撑，以及是否在计划完成前过早结束”。

必须遵守：
1. 只输出 JSON。
2. 如果当前回答已经能被 `Messages`、`Session Summary`、`Compacted Session Context`、`Recent Content References` 或现有 `Loop Events` 充分支持，并且没有在计划完成前提前结束，输出 `accept`。
3. 如果当前回答明显缺乏证据、遗漏了应当检索的历史内容、当前只掌握 preview/snippet 但仍不足以支撑回答，或当前计划显然还没完成却试图直接结束，输出 `retry`。
4. `retry` 只表示应让决策模型重新判断是否要继续当前步骤或发起重规划，不要直接指定固定工具。

JSON schema:
{
  "verdict": "accept" | "retry",
  "reason": "string"
}
""".strip()


def build_plan_system_prompt(
    workspace_docs: str,
    tool_prompt_docs: str,
    tool_names: tuple[str, ...],
) -> str:
    return f"{workspace_docs}\n\n{_build_plan_instruction(tool_names, tool_prompt_docs)}"


def build_decision_system_prompt(
    workspace_docs: str,
    tool_prompt_docs: str,
    tool_names: tuple[str, ...],
) -> str:
    return f"{workspace_docs}\n\n{_build_decision_instruction(tool_names, tool_prompt_docs)}"


def build_decision_review_system_prompt(workspace_docs: str) -> str:
    return f"{workspace_docs}\n\n{DECISION_REVIEW_INSTRUCTION}"
