# Novel Agent V2

封闭式小说 Agent 原型，面向“小说域对话 + 章节压缩 + 记忆检索”场景。

当前版本坚持两个边界：

- 只服务小说相关任务
- 小说域外统一返回 `无法回答`

## V2 更新

相较于最初的 V1，V2 做了 6 个核心升级：

- 从“单步 JSON 路由”升级为“多步 agent loop”
  - 决策模型不再只做一次直答/调 tool 判断，而是可以在一轮中多步思考、调用工具、接收 observation、再继续决策
- 从“单工具压缩”升级为“压缩 + memory tools”
  - 新增 `memory_search`
  - 新增 `memory_get`
  - 当前 tool 集合为：`compress_chapter`、`memory_search`、`memory_get`
- 从“临时会话”升级为“可恢复 transcript”
  - 会话事件按 JSONL 追加到 `runtime/sessions/<session_id>.jsonl`
  - 会话索引记录在 `runtime/sessions/sessions.json`
- 新增 heartbeat 维护层
  - `turn heartbeat`：每轮结束后刷新 session summary、标脏 memory 维护状态
  - `idle heartbeat`：按机会性维护 daily memory / long-term memory
- Memory 从“只注入 Markdown”升级为“Markdown + memory tools”
  - Workspace 仍保留 `memory.md` 和 `workspace/memory/YYYY-MM-DD.md`
  - 但决策模型现在可以显式检索和读取 memory，而不是假设它们被完整注入
- 前端 Gradio Demo 重做
  - 左侧专注聊天
  - 右侧显示运行状态、模型 loop 流程、本轮 memory 写入
  - 聊天区支持固定高度滚动，适合多轮对话

## 当前能力

### 1. 小说域多轮对话

- 决策模型默认使用本地 `Qwen/Qwen3-4B`
- 允许的直接交流包括：
  - 剧情讨论
  - 人物关系分析
  - 设定理解
  - 压缩需求澄清
- 小说域外请求统一返回 `无法回答`

### 2. 章节压缩

- 压缩 tool 使用微调后的压缩模型
- 默认配置路径见 `novel_agent/config.py`
- 当输入明显表达“压缩小说/章节/原文”的含义时，会优先路由到 `compress_chapter`
- 压缩目标是：
  - 缩短篇幅
  - 保持事实不变
  - 保持人物关系不变
  - 保持叙事顺序不变

### 3. Memory 检索与读取

- `memory_search`
  - 用于检索长期记忆和 daily memory 中的相关片段
- `memory_get`
  - 用于读取指定 memory 文档
  - 支持目标如：
    - `long_term`
    - `daily_latest`
    - `today`
    - `yesterday`
    - `daily:YYYY-MM-DD`

### 4. 会话记录与记忆维护

- 完整会话事件：
  - `runtime/sessions/<session_id>.jsonl`
- 会话索引：
  - `runtime/sessions/sessions.json`
- 长期记忆：
  - `workspace/memory.md`
- 每日记忆：
  - `workspace/memory/YYYY-MM-DD.md`
- 维护机制：
  - `turn heartbeat`
  - `idle heartbeat`

## 运行结构

- `novel_agent/app.py`
  - Gradio UI 入口
- `novel_agent/controller.py`
  - 多步 agent loop、工具调度、受控回复
- `novel_agent/backends/decision.py`
  - 决策模型加载与推理
- `novel_agent/backends/compression.py`
  - 压缩模型加载与推理
- `novel_agent/workspace.py`
  - workspace 文档与 memory 访问
- `novel_agent/memory.py`
  - session transcript 与会话恢复
- `novel_agent/heartbeat.py`
  - heartbeat 调度
- `novel_agent/maintenance.py`
  - summary / daily memory / long-term memory 维护

## Workspace 设计

当前 workspace 采用 Markdown 作为可编辑真相源：

- `workspace/agent.md`
  - 封闭能力边界
- `workspace/tools.md`
  - 工具声明
- `workspace/identity.md`
  - 身份设定
- `workspace/soul.md`
  - 风格与行为倾向
- `workspace/user.md`
  - 用户偏好
- `workspace/memory.md`
  - 长期记忆
- `workspace/memory/YYYY-MM-DD.md`
  - 每日记忆
  - 运行时本地生成，默认不纳入 Git 版本跟踪

## 目录

- `novel_agent/`
  - 核心代码
- `workspace/`
  - Agent 文档与 memory
- `tests/`
  - 单元与集成测试
- `runtime/`
  - 本地运行时数据目录（已加入 `.gitignore`）

## 启动

安装依赖后运行：

```bash
cd /home/ubuntu/code/novel_agent
python -m novel_agent
```

默认会启动一个本地 Gradio Demo。

## 测试

```bash
cd /home/ubuntu/code/novel_agent
python -m pytest -q tests
```

## 当前默认配置

- 决策模型路径：
  - `/home/ubuntu/code/Qwen/Qwen3-4B`
- 压缩模型路径：
  - `/home/ubuntu/code/qwen/output/Qwen3-4B_cot_beta_4/checkpoint-1334`
- 决策模型默认参数：
  - `temperature = 0.7`
  - `max_new_tokens = 32768`
- 压缩模型默认参数：
  - `temperature = 0.2`
  - `max_new_tokens = 2800`

## 当前限制

- 仍然是封闭式小说 agent，不是通用助手
- 当前唯一“内容生成型 tool”仍是 `compress_chapter`
- `memory_search` / `memory_get` 是辅助工具，不负责生成章节内容
- 没有联网搜索
- 没有向量数据库
- 没有多 agent 协作

## 版本说明

本次提交将仓库从 `Novel Agent V1` 更新为 `Novel Agent V2`，重点标志是：

- 多步 loop
- transcript 升级
- memory tools
- heartbeat
- 新版 UI
