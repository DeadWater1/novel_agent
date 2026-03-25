# Novel Agent V1

封闭式小说 Agent 原型。

当前版本特性：

- 决策模型固定为 `/home/ubuntu/code/Qwen/Qwen3-4B`
- 压缩 tool model 逻辑对齐 `/home/ubuntu/code/qwen/infer.py`
- 仅服务小说域
- 小说域外统一返回 `无法回答`
- 当前唯一可执行 tool 为 `compress_chapter`
- 采用 Markdown workspace + memory 分层

## 目录

- `novel_agent/`：核心代码
- `workspace/`：Agent 文档与 memory
- `tests/`：单元与集成测试

## 运行

安装依赖后可启动：

```bash
python -m novel_agent
```

如果缺少 `gradio`、`transformers` 或 `modelscope`，应用会给出明确错误或后端健康状态异常提示。

## 依赖

参见 `requirements.txt`。
