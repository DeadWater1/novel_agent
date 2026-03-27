# Tools

## compress_chapter

- 名称：`compress_chapter`
- 作用：对小说章节进行压缩
- 输入：
  - `raw_text`: 原始章节文本，必填
- 说明：生成参数由系统配置统一管理，决策模型不应尝试覆盖
- 输出：
  - `compressed_text`
  - `thinking`（仅调试可见）

## memory_search

- 名称：`memory_search`
- 作用：检索长期记忆和日记式记忆中的相关小说信息
- 输入：
  - `query`: 查询关键词或问题，必填
  - `max_results`: 可选，返回条数上限
- 输出：
  - 匹配的记忆片段列表

## memory_get

- 名称：`memory_get`
- 作用：读取指定 target 对应的完整内容或完整摘要
- 输入：
  - `target`: 必填，可选值包括 `long_term`、`daily_latest`、`today`、`yesterday`、`daily:YYYY-MM-DD`、`session:...`、`session_compact:...`、`content_ref:latest`
  - `delivery_mode`: 可选，默认 `observe`；若明确要把全文直接交付给用户，可显式传 `deliver`
- 输出：
  - target 对应的完整内容或摘要

## embedding_similarity

- 名称：`embedding_similarity`
- 作用：使用本地 embedding 模型比较 query 与候选文本的语义相似度
- 输入：
  - `query`: 必填
  - `text`: 可选，单条候选文本
  - `texts`: 可选，多条候选文本列表
  - `top_k`: 可选，仅保留最高分前 k 条
- 输出：
  - 相似度分数列表

## 规则

- 当前版本工具共四个：`compress_chapter`、`memory_search`、`memory_get`、`embedding_similarity`
- 如果请求属于小说域，但不需要工具，可以直接回复
- 如果需要记忆信息，可以选择 `memory_search` 或 `memory_get`
- 如果已经拿到若干候选文本但还需要比较语义接近度，可以使用 `embedding_similarity`
