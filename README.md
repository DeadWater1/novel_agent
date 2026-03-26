# Novel Agent V3

封闭式小说 Agent 的 V3 版本，面向“小说域对话 + 章节压缩 + 可检索会话记忆”场景。

当前版本仍坚持两个边界：

- 只服务小说相关任务
- 小说域外统一返回 `无法回答`

## V3 更新

相较于 `v2.0.0`，V3 的核心变化不是只加几个工具，而是把整体运行方式从“控制器主导”改成了“模型主导 + 上下文引擎 + 分层记忆检索”。

### 1. 从“控制器硬路由”升级为“模型主导的多步 agent loop”

- 删除了 V2 中残留的业务型硬路由与强制捷径
  - 不再由 controller 直接替模型决定“最近压缩结果”“压缩次数”“第一次/第二次压缩内容”
- 决策模型成为真正的主入口
  - 先理解用户需求
  - 再决定 `direct_reply`、`memory_search`、`memory_get` 或 `compress_chapter`
- 新增一次隐藏式 `review` 回合
  - 当首答证据不足时，允许模型自我复审并重试，而不是靠关键词规则强行纠偏

### 2. 从“短窗口上下文”升级为“32K 上下文引擎”

- 新增 `ContextEngine`
  - 统一负责上下文装配
  - 统一负责 token 估算
  - 统一负责 pruning / compaction / memory flush
- 决策模型输入预算固定为 `32K`
  - `decision_input_token_budget = 32768`
  - `decision_output_max_new_tokens = 4096`
- 上下文管理改成三层
  - `recent raw messages`
  - `Compacted Session Context`
  - `Recent Content References`
- 新增预压缩 memory flush
  - 接近阈值时先静默落 durable facts
  - 再进入持久 compaction

### 3. 从“弱 memory_search”升级为“分层历史检索”

- `memory_search` 现在区分：
  - `lookup`
  - `recap`
- 检索范围支持：
  - `current_session`
  - `history_sessions`
  - `time_window`
- 检索源升级为多层混合
  - 当前 session 原文
  - compact summary / compression ledger
  - 历史 session transcript
  - workspace Markdown memory
- 对压缩历史新增了统一账本层
  - 每次压缩都有 `target`
  - 每次压缩都可通过 `memory_get` 回溯全文

### 4. 从“memory_get 返回片段”升级为“默认全文交付”

- `memory_get` 默认直接交付完整正文
- 模型侧只看短 observation，不再把全文重新塞回 prompt
- 新增 `Recent Content References`
  - 让模型在后续追问里继续读取上一次内容
  - 支持 `content_ref:latest`
- 针对压缩历史的读取链路进一步收紧
  - `Compacted Session Context` 里的压缩记录只展示索引信息，不再把 preview 当正文
  - lookup 场景中，更具体的 `session_compact:...#compression_history:N` 会优先于 plain `session_compact:SESSION_ID`

### 5. 从“单一决策模型”升级为“决策 + 摘要后端”

- 新增 `LocalSummaryBackend`
  - 专门负责会话 compaction
  - 默认复用本地 `Qwen3-14B`
- 新增运行时产物：
  - `runtime/transcripts/<session_id>/transcript_XXXX.jsonl`
  - `runtime/compactions/<session_id>.json`

### 6. UI 与交互体验继续重做

- 首页标题升级为 `Novel Agent V3`
- 聊天区更宽、更高，长对话可读性更好
- 输入框发送后立即清空
- 处理过程中显示 `正在处理...`
- 右侧 `RUNTIME` 和 `LOOP` 面板改成独立滚动
- 旧会话 compaction 改为后台补做，不再阻塞 UI 首屏启动

## 相比 V2 的核心增删改

### 新增

- `novel_agent/context_engine.py`
  - 上下文引擎
- `novel_agent/compaction.py`
  - 会话压缩、compression ledger、transcript snapshot
- `novel_agent/search_utils.py`
  - hybrid search / MMR / snippet utilities
- `novel_agent/backends/summary.py`
  - LLM 摘要后端
- `Recent Content References`
- `Context Report` 内部数据结构
- `session_compact:*` / `content_ref:*` 目标解析

### 重构

- `novel_agent/controller.py`
  - 从“路由器 + 规则集”改成“标准 agent loop + tool execution + review”
- `novel_agent/backends/decision.py`
  - 改成 32K 输入预算与 GPU/bf16 优先加载
- `novel_agent/workspace.py`
  - 检索逻辑与 session memory 访问增强
- `novel_agent/app.py`
  - UI、启动流程、滚动布局、处理中占位等整体重构
- `novel_agent/heartbeat.py`
  - 接入 compaction / memory flush 维护

### 删除或替换

- 删除 V2 残留的 controller 业务硬路由
  - 最近压缩捷径
  - 压缩次数捷径
  - 第一次/第二次压缩结果直返捷径
- 替换 V2 的“小上下文 + 人工猜范围”
  - 现在改成模型显式选择检索范围与 search mode
- 替换“正文常被截断喂回模型”的旧行为
  - 现在全文直接交付给用户，模型只接收 observation 节选

## 当前能力

### 1. 小说域多轮对话

- 决策模型默认使用本地 `Qwen3-14B`
- 默认 GPU + bf16 优先加载
- 支持剧情讨论、人物关系分析、设定理解、压缩前澄清

### 2. 章节压缩

- `compress_chapter` 使用本地压缩模型
- 压缩目标仍然保持：
  - 缩短篇幅
  - 保持事实不变
  - 保持人物关系不变
  - 保持叙事顺序不变

### 3. 记忆检索与回读

- `memory_search`
  - 既可做 `lookup`，也可做 `recap`
- `memory_get`
  - 默认读取并交付完整正文
  - 支持：
    - `long_term`
    - `daily_latest`
    - `today`
    - `yesterday`
    - `daily:YYYY-MM-DD`
    - `session:...`
    - `session_compact:...`
    - `content_ref:latest`

### 4. 会话压缩与归档

- Transcript snapshot：
  - `runtime/transcripts/`
- Compact artifact：
  - `runtime/compactions/`
- Session transcript：
  - `runtime/sessions/`

## 运行结构

- `novel_agent/app.py`
  - Gradio UI 与启动入口
- `novel_agent/controller.py`
  - agent loop、工具执行、review、memory_get 终结回复
- `novel_agent/context_engine.py`
  - 上下文预算、pruning、auto recall、memory flush、compaction 装配
- `novel_agent/compaction.py`
  - compact artifact、compression ledger、transcript snapshot
- `novel_agent/backends/decision.py`
  - 决策模型加载与推理
- `novel_agent/backends/summary.py`
  - 会话摘要模型
- `novel_agent/backends/compression.py`
  - 压缩模型加载与推理
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
- 摘要模型路径：
  - `/home/ubuntu/code/qwen/Qwen/Qwen3-14B`
- 压缩模型路径：
  - `/home/ubuntu/code/qwen/output/Qwen3-4B_cot_beta_4/checkpoint-1334`
- 决策预算：
  - `decision_input_token_budget = 32768`
  - `decision_output_max_new_tokens = 4096`
- 上下文阈值：
  - `context_memory_flush_soft_threshold = 28672`
  - `context_pruning_soft_budget = 30720`
  - `context_auto_compact_token_threshold = 31744`

## 当前限制

- 仍然是封闭式小说 Agent，不是通用助手
- 当前唯一内容生成型 tool 仍是 `compress_chapter`
- 没有联网搜索
- 没有向量数据库
- 没有多 agent 协作

## 版本说明

- `v1.0.0`
  - 初始单步版本
- `v2.0.0`
  - 多步 loop、memory tools、heartbeat、Gradio UI
- `v3.0.0`
  - 模型主导 loop、32K 上下文引擎、分层 memory retrieval、compaction/runtime artifacts、全文交付型 memory_get
