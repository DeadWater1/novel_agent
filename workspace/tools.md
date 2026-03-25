# Tools

## compress_chapter

- 名称：`compress_chapter`
- 作用：对小说章节进行压缩
- 输入：
  - `raw_text`: 原始章节文本，必填
  - `max_new_tokens`: 可选
  - `temperature`: 可选
  - `seed`: 可选
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
- 作用：获取指定 memory 文档原文
- 输入：
  - `target`: 必填，可选值包括 `long_term`、`daily_latest`、`today`、`yesterday`、`daily:YYYY-MM-DD`
- 输出：
  - 指定 memory 文档内容

## 规则

- 当前版本工具共三个：`compress_chapter`、`memory_search`、`memory_get`
- 如果请求属于小说域，但不需要工具，可以直接回复
- 如果需要记忆信息，优先使用 `memory_search` 或 `memory_get`
- 如果用户输入中明显表达了“压缩小说/章节/原文”的含义，应优先直接路由到 `compress_chapter`
