# arXiv QA CLI

一个最小工具：从命令行输入多组 `arXiv URL + query`，自动下载 PDF，使用 `PyMuPDF4LLM` 提取文本，再调用 Azure OpenAI Responses API 输出答案。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 用法

```bash
python3 arxiv_qa.py \
  --task "https://arxiv.org/abs/1706.03762" "What is the main contribution?" \
  --task "https://arxiv.org/abs/2401.12345" "Summarize the experimental setup."
```

默认会并发处理多组任务，默认并发数是 `16`，可用 `--workers` 控制：

```bash
python3 arxiv_qa.py \
  --workers 4 \
  --task "https://arxiv.org/abs/1706.03762" "What is the main contribution?" \
  --task "https://arxiv.org/abs/1810.04805" "Summarize the main idea."
```

也可以直接传 PDF URL：

```bash
python3 arxiv_qa.py \
  --task "https://arxiv.org/pdf/1706.03762.pdf" "Give me a short summary."
```

输出 JSON：

```bash
python3 arxiv_qa.py \
  --format json \
  --task "https://arxiv.org/abs/1706.03762" "List the core ideas."
```

## 可选参数

- `--model`：默认 `gpt-5.4-2026-03-05`
- `--api-key`：默认读取内置值，也可通过环境变量 `AZURE_OPENAI_API_KEY` 覆盖
- `--endpoint`：默认 `https://aidp.bytedance.net/api/modelhub/online/responses`
- `--api-version`：默认 `2024-03-01-preview`
- `--logid`：默认 `arxiv-qa-cli`
- `--max-output-tokens`：传给 Responses API 的输出上限，默认 `32768`
- `--max-chars`：限制送入模型的最大字符数，默认 `120000`
- `--workers`：并发处理任务数，默认 `16`

## 说明

- 每个 `--task` 对应一条结果，多个任务会并发执行。
- 如果 PDF 下载失败、解析失败或模型调用失败，会在对应结果中返回 `ERROR: ...`。
