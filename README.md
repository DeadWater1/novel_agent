# Novel Agent V4

封闭式小说 Agent 的 V4 版本，面向“小说域对话 + 章节压缩 + 可检索会话记忆 + 计划化多步执行”场景。

当前版本继续坚持两个边界：

- 只服务小说相关任务
- 小说域外统一返回 `无法回答`

## V4 更新

相较于 `v3.0.0`，V4 的核心变化不是继续堆工具，而是把 agent 主链从“单步决策 loop”升级成了“显式计划 + 按 step 执行”的模型主导流程。

### 1. 从“单步决策”升级为“先计划、再执行”

- 每个用户问题都会先触发一次 `plan_turn(...)`
  - 决策模型先输出 `ExecutionPlanOutput`
  - 计划里显式列出 `steps`
- controller 不再直接一轮只产出一个动作
  - 现在会维护当前 step
  - 会维护已完成 step
  - 会按计划逐步推进
- 执行中允许一次 `plan_update`
  - 如果模型判断原计划不合理，可以重规划剩余步骤

### 2. 保留 review，但职责升级为“证据检查 + 计划完成度检查”

- V4 仍然保留一次隐藏 review
- review 不再只看“首答证据够不够”
- 现在还会检查：
  - 当前回答是否在计划完成前过早结束
  - 当前 step 是否真的完成
- `retry` 时仍然不会由 controller 强行指定工具，仍然交回给模型自己重新判断

### 3. `memory_get` 从“默认直接交付”改成“默认 observe”

- `memory_get` 默认值已改为：
  - `delivery_mode="observe"`
- 这意味着：
  - 默认先读内容
  - 把 observation 节选回给模型
  - 是否要把全文真正交付给用户，由模型显式决定
- 只有模型明确传：
  - `delivery_mode="deliver"`
  才会把全文直接返回给前端

这一步是为了更自然地支持多步计划，例如：

1. 先 `memory_get(observe)` 读取最近一次压缩结果  
2. 再 `direct_reply` 返回“总共有多少字”

### 4. Prompt 进一步去路径化，强化模型自主决策

- 删除了 prompt 中那类“先 recap / lookup”“证据不足就 call_tool”的路径引导语
- 不再让 prompt 预先替模型规划动作顺序
- 新增单独的 planning prompt
  - 只负责拆分子任务
  - 不负责直接给最终答案
- 决策 prompt 改成“执行当前 step”

### 5. 去掉 auto recall，是否检索交给模型自己决定

- `ContextEngine` 不再在每轮决策前自动替模型做一次 `current_session` recall
- 现在：
  - 查不查记忆
  - 查哪一层历史
  - 是否需要 `memory_search`
  都由决策模型自己决定

### 6. UI 同步升级为“计划优先”

- 右侧 `LOOP` 面板新增：
  - `Plan Created`
  - `Plan Updated`
  - `Step Completed`
- 右侧“决策模型输出”窗口现在分开展示：
  - `Planner Output`
  - `Decision Output`
  - `Review Output`
- 首页标题升级为 `Novel Agent V4`

## 相比 V3 的核心增删改

### 新增

- `PlanStep`
- `ExecutionPlanOutput`
- `decision_backend.plan_turn(...)`
- planner 原始输出展示
- plan 事件：
  - `plan_created`
  - `plan_updated`
  - `plan_step_completed`

### 重构

- `novel_agent/controller.py`
  - 从“单步决策 + 工具执行”重构为“计划阶段 + 执行阶段”
- `novel_agent/backends/decision.py`
  - 新增 planner 调用
  - 保留 planner / decision / review 三类原始输出
- `novel_agent/prompts.py`
  - 拆成 planning prompt、execution prompt、review prompt
- `novel_agent/app.py`
  - Loop UI 改成可视化计划执行
  - 决策输出面板增加 planner 原始输出

### 删除或替换

- 删除 prompt 中对工具路径的强引导句
- 删除 auto recall 的预检索行为
- 替换 V3 中 `memory_get` 默认直接交付全文的行为
  - 现在默认 `observe`
  - 显式 `deliver` 才会全文直返

## 当前能力

### 1. 小说域多轮对话

- 决策模型默认使用本地 `Qwen3-14B`
- 默认 GPU + bf16 优先加载
- 已支持显式 plan -> execute 的多步 agent loop

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
    - `daily_latest`
    - `today`
    - `yesterday`
    - `daily:YYYY-MM-DD`
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
- Session transcript：
  - `runtime/sessions/`

## 运行结构

- `novel_agent/app.py`
  - Gradio UI 与启动入口
- `novel_agent/controller.py`
  - planning / execution loop、工具执行、review、memory_get 行为
- `novel_agent/context_engine.py`
  - 上下文预算、pruning、memory flush、compaction 装配
- `novel_agent/compaction.py`
  - compact artifact、compression ledger、transcript snapshot
- `novel_agent/embedding_index.py`
  - 本地 embedding shard 缓存与查询
- `novel_agent/backends/decision.py`
  - planner / decision / review 模型加载与推理
- `novel_agent/backends/compression.py`
  - 压缩模型加载与推理
- `novel_agent/backends/embedding.py`
  - embedding 模型加载与相似度计算
- `novel_agent/workspace.py`
  - workspace 文档、Markdown memory、memory access
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
- `workspace/memory/YYYY-MM-DD.md`

### Runtime

- `runtime/sessions/`
- `runtime/transcripts/`
- `runtime/compactions/`
- `runtime/embeddings/`

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
- loop 上限：
  - `agent_max_loop_steps = 6`
- 上下文阈值：
  - `context_memory_flush_soft_threshold = 28672`
  - `context_pruning_soft_budget = 30720`
  - `context_auto_compact_token_threshold = 31744`

## 当前限制

- 仍然是封闭式小说 Agent，不是通用助手
- 当前仍然只有单 Agent 执行，不做多 agent 并行
- 没有联网搜索
- 没有外部向量数据库
- 计划是 turn-local 的，不跨多个用户回合持久延续

## 版本说明

- `v1.0.0`
  - 初始单步版本
- `v2.0.0`
  - 多步 loop、memory tools、heartbeat、Gradio UI
- `v3.0.0`
  - 模型主导 loop、32K 上下文引擎、分层 memory retrieval、compaction/runtime artifacts
- `v4.0.0`
  - 显式 planner、多步计划执行、单次重规划、review 检查计划完成度、`memory_get` 默认 observe、planner UI 可视化
