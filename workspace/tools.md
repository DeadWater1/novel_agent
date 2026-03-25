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

## 规则

- 当前版本仅有这一个工具
- 如果请求属于小说域，但不需要工具，可以直接回复
- 如果请求需要工具，只能调用 `compress_chapter`
- 如果用户输入中明显表达了“压缩小说/章节/原文”的含义，应优先直接路由到 `compress_chapter`
