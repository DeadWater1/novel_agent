# Novel Agent V5

封闭式小说 Agent 的 V5 版本，面向“小说域对话 + 章节压缩 + 可检索会话记忆 + 计划化多步执行”场景。

当前版本继续坚持两个边界：

- 只服务小说相关任务
- 小说域外统一返回 `无法回答`

## V5 更新

相较于 `v4.0.0`，V5 的核心变化是把 agent 从“能规划、能调工具”继续推进到“更稳地执行、更稳地收口、底层更易扩展”。

### 1. Toolbox 升级为可扩展框架

- 工具系统不再把工具名写死在 prompt / schema / controller
- `ToolRegistry` 现在成为工具元数据单一真源
- 每个工具拥有独立 `Args model + handler + prompt_doc`
- 新增第 5 个工具时，不需要再改 controller 分支或 prompt 常量

### 2. 自动记忆改为结构化主存储

- 人工长期规则继续常驻：
  - `agent.md`
  - `identity.md`
  - `user.md`
  - `soul.md`
- 自动历史记忆不再默认注入 prompt
- 自动 memory 迁移到：
  - `runtime/structured_memory/context.json`
  - `runtime/structured_memory/facts.jsonl`
  - `runtime/structured_memory/digests/YYYY-MM-DD.md`
- `memory_search` / `memory_get` 继续由模型自行决定是否调用

### 3. 提前结束保护与最终收口补齐

- 未完成 plan 时，`direct_reply` 不允许提前结束整轮
- 当前 plan 后面还有步骤时，`terminal tool` 会降级为 observation，而不是直接把结果交付用户
- 当本轮已经发生过工具调用，且 plan 已执行到末尾时，controller 会自动进入一次 `final synthesis`
- `final synthesis` 会把：
  - 用户原问题
  - 当前上下文
  - 已完成步骤
  - 结构化工具证据
  再喂给 decision，强制生成最终 `direct_reply`
- 若第一次收口失败，会自动再试一次，避免直接落入 `plan_exhausted`

### 4. 证据回喂改成“结构化证据优先”

- 工具结果不会默认把所有全文重新塞回 decision
- 优先回喂：
  - `memory_get` 的 `target / resolved_target / source_path / content_length / preview`
  - `memory_search` 的 `target / score / snippet / summary_preview`
  - `embedding_similarity` 的 `candidate_count / top_score / best_match_index / exact_match`
- 上下文超预算时，优先裁剪旧 loop events 和长 preview，只对本轮相关证据做局部摘要

### 5. 生成链支持本地 vLLM

- 文本生成链支持：
  - `local`
  - `vllm`
- `decision / review / summary / compression` 可以走本地 vLLM
- embedding 仍保持本地独立实现
- 当前默认 review 模式为 `risk_gated`
- 当前 loop 上限提升到 `16`

## 相比 V4 的核心增删改

### 新增

- 可扩展 toolbox handler / registry 框架
- structured memory 主存储
- final synthesis 收口阶段
- vLLM generation backend
- plan 执行保护事件：
  - `premature_direct_reply_blocked`
  - `terminal_tool_deferred`
  - `final_synthesis_started`
  - `final_synthesis_retry`
  - `final_synthesis_completed`
  - `final_synthesis_failed`

### 重构

- `novel_agent/controller.py`
  - 从“按计划执行”升级为“按计划执行 + 工具后强制收口”
- `novel_agent/backends/decision.py` / `novel_agent/backends/vllm_backend.py`
  - 支持 final synthesis 场景的最终答复生成
- `novel_agent/prompts.py`
  - 强化判断型问题必须保留最终回答出口
- `novel_agent/workspace.py`
  - 自动 memory 改为 structured memory 读取与检索

### 删除或替换

- 移除 Ollama 方案，统一改为 `local / vllm`
- daily markdown 不再是自动 memory 真源
- 替换“工具执行完即自然结束”的链路
  - 现在必须经过最终收口或终局工具结束

## 当前能力

### 1. 小说域多轮对话

- 决策模型默认使用本地 `Qwen3-14B`
- 默认 GPU + bf16 优先加载
- 已支持显式 plan -> execute -> final synthesis 的多步 agent loop

### 2. 章节压缩

- `compress_chapter` 使用本地压缩模型
- 默认启用 thinking
- 聊天框仅显示 answer，不显示 thinking
- 当前默认压缩预算：
  - `compression_max_new_tokens = 3584`
  - `compression_answer_reserved_tokens = 1536`

### 3. 记忆检索与回读

- `memory_search`
  - 支持 `lookup`
  - 支持 `recap`
  - 支持 `current_session / history_sessions / time_window`
- `memory_get`
  - 默认 `observe`
  - 显式 `deliver` 时才全文交付
  - 支持：
    - `long_term`
    - `context:user_preferences`
    - `context:story_constraints`
    - `context:open_loops`
    - `fact:MEMORY_ID`
    - `digest:YYYY-MM-DD`
    - `daily_latest / today / yesterday / daily:YYYY-MM-DD`（兼容映射）
    - `session:...`
    - `session_compact:...`
    - `content_ref:latest`

### 4. 向量检索与预索引

- `memory_search` 已改为本地 embedding 相似度检索
- embedding 模型：
  - `/home/ubuntu/code/qwen/Qwen/Qwen3-Embedding-0.6B`
- embedding 窗口：
  - `32768`
- 稳定历史层支持空闲时预索引：
  - workspace memory
  - compaction chunks
  - history session chunks

### 5. 会话压缩与归档

- Transcript snapshot：
  - `runtime/transcripts/`
- Compact artifact：
  - `runtime/compactions/`
- Embedding shards：
  - `runtime/embeddings/`
- Structured memory：
  - `runtime/structured_memory/`
- Session transcript：
  - `runtime/sessions/`

## 运行结构

- `novel_agent/controller.py`
  - planning / execution loop、工具执行、review、final synthesis
- `novel_agent/context_engine.py`
  - 上下文预算、pruning、memory flush、compaction 装配
- `novel_agent/toolbox.py`
  - 工具 handler、参数模型与执行分发
- `novel_agent/compaction.py`
  - compact artifact、compression ledger、transcript snapshot
- `novel_agent/embedding_index.py`
  - 本地 embedding shard 缓存与查询
- `novel_agent/backends/decision.py`
  - planner / decision / review 模型加载与推理
- `novel_agent/backends/vllm_backend.py`
  - vLLM 本地推理后端
- `novel_agent/backends/compression.py`
  - 压缩模型加载与推理
- `novel_agent/backends/embedding.py`
  - embedding 模型加载与相似度计算
- `novel_agent/workspace.py`
  - workspace 文档、structured memory、memory access
- `novel_agent/memory.py`
  - session transcript 与会话恢复
- `novel_agent/heartbeat.py`
  - turn / idle heartbeat

## Workspace 与 Runtime

### Workspace

- `workspace/agent.md`
- `workspace/tools.md`
- `workspace/identity.md`
- `workspace/soul.md`
- `workspace/user.md`
- `workspace/memory.md`

### Runtime

- `runtime/sessions/`
- `runtime/transcripts/`
- `runtime/compactions/`
- `runtime/embeddings/`
- `runtime/structured_memory/`

## 启动

```bash
cd /home/ubuntu/code/novel_agent
python -m novel_agent
```

## 测试

```bash
cd /home/ubuntu/code/novel_agent
python -m pytest -q tests
```

## 当前默认配置

- 决策模型路径：
  - `/home/ubuntu/code/qwen/Qwen/Qwen3-14B`
- embedding 模型路径：
  - `/home/ubuntu/code/qwen/Qwen/Qwen3-Embedding-0.6B`
- 摘要模型路径：
  - `/home/ubuntu/code/qwen/Qwen/Qwen3-14B`
- 压缩模型路径：
  - `/home/ubuntu/code/qwen/output/Qwen3-4B_cot_beta_4/checkpoint-1334`
- 决策预算：
  - `decision_input_token_budget = 32768`
  - `decision_output_max_new_tokens = 4096`
- vLLM：
  - `vllm_gpu_memory_utilization = 0.5`
  - `vllm_max_model_len = 32768`
- loop 上限：
  - `agent_max_loop_steps = 16`
- 上下文阈值：
  - `context_memory_flush_soft_threshold = 28672`
  - `context_pruning_soft_budget = 30720`
  - `context_auto_compact_token_threshold = 31744`

## 当前限制

- 仍然是封闭式小说 Agent，不是通用助手
- 当前仍然只有单 Agent 执行，不做多 agent 并行
- 没有联网搜索
- 没有外部向量数据库
- 计划仍然是 turn-local 的，不跨多个用户回合持久延续
- final synthesis 仍依赖 decision 模型生成最终结论，不做 controller 模板化保底

## 版本说明

- `v1.0.0`
  - 初始单步版本
- `v2.0.0`
  - 多步 loop、memory tools、heartbeat、Gradio UI
- `v3.0.0`
  - 模型主导 loop、32K 上下文引擎、分层 memory retrieval、compaction/runtime artifacts
- `v4.0.0`
  - 显式 planner、多步计划执行、单次重规划、review 检查计划完成度、`memory_get` 默认 observe
- `v5.0.0`
  - 可扩展 toolbox、structured memory、提前结束保护、工具后强制收口、vLLM backend、loop 上限 16
