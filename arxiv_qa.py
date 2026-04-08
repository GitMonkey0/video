#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
import pymupdf4llm


DEFAULT_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY",
    "YDbc4R5T3MLZxeuDq2fQTLLtCfPP0TVC_GPT_AK",
)
DEFAULT_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://aidp.bytedance.net/api/modelhub/online/responses",
)
DEFAULT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")
DEFAULT_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5.4-2026-03-05")
DEFAULT_LOGID = os.getenv("AZURE_OPENAI_LOGID", "arxiv-qa-cli")
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("AZURE_OPENAI_MAX_OUTPUT_TOKENS", "32768"))


@dataclass
class Task:
    source_url: str
    pdf_url: str
    query: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download arXiv papers, parse PDFs with PyMuPDF4LLM, and answer queries with Azure OpenAI."
    )
    parser.add_argument(
        "--task",
        nargs=2,
        action="append",
        metavar=("URL", "QUERY"),
        required=True,
        help='Repeatable. Example: --task "https://arxiv.org/abs/1706.03762" "What is the main contribution?"',
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--logid", default=DEFAULT_LOGID)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens sent to the Responses API.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=120000,
        help="Maximum number of extracted characters sent to the model for each paper.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Maximum number of tasks processed concurrently.",
    )
    return parser.parse_args()


def normalize_arxiv_url(url: str) -> str:
    stripped = url.strip()
    if stripped.endswith(".pdf"):
        return stripped

    abs_match = re.match(r"^https?://arxiv\.org/abs/([^?#]+)", stripped)
    if abs_match:
        paper_id = abs_match.group(1)
        return f"https://arxiv.org/pdf/{paper_id}.pdf"

    pdf_match = re.match(r"^https?://arxiv\.org/pdf/([^?#]+)$", stripped)
    if pdf_match:
        paper_id = pdf_match.group(1)
        if paper_id.endswith(".pdf"):
            return stripped
        return f"https://arxiv.org/pdf/{paper_id}.pdf"

    raise ValueError(f"Unsupported arXiv URL: {url}")


def build_tasks(raw_tasks: list[list[str]]) -> list[Task]:
    tasks: list[Task] = []
    for source_url, query in raw_tasks:
        tasks.append(
            Task(
                source_url=source_url,
                pdf_url=normalize_arxiv_url(source_url),
                query=query,
            )
        )
    return tasks


def download_pdf(url: str) -> Path:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; arxiv-qa-cli/1.0)",
        },
    )
    with urllib.request.urlopen(request) as response:
        if response.status != 200:
            raise RuntimeError(f"Failed to download PDF from {url}: HTTP {response.status}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.read())
            return Path(tmp_file.name)


def extract_markdown(pdf_path: Path) -> str:
    markdown = pymupdf4llm.to_markdown(str(pdf_path))
    if not markdown or not markdown.strip():
        raise RuntimeError(f"No text extracted from PDF: {pdf_path}")
    return markdown


def trim_content(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


def create_client(api_key: str, endpoint: str, api_version: str) -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def response_to_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    texts: list[str] = []
    for item in _value(response, "output", []) or []:
        if _value(item, "type") != "message":
            continue
        for content in _value(item, "content", []) or []:
            content_type = _value(content, "type")
            if content_type in {"output_text", "text"}:
                text = _value(content, "text", "")
                if text:
                    texts.append(text)

    if texts:
        return "\n".join(texts).strip()

    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    return str(response)


def answer_query(
    client: openai.AzureOpenAI,
    model: str,
    logid: str,
    max_output_tokens: int,
    paper_url: str,
    paper_text: str,
    query: str,
) -> str:
    prompt = (
        "You are answering a question about an academic paper.\n"
        "Use only the supplied paper content.\n"
        "If the answer is not supported by the paper text, say that clearly.\n\n"
        f"Paper URL: {paper_url}\n\n"
        f"Question: {query}\n\n"
        "Paper content:\n"
        f"{paper_text}"
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
        extra_headers={
            "X-TT-LOGID": logid,
        },
    )
    return response_to_text(response)


def process_task(
    task: Task,
    api_key: str,
    endpoint: str,
    api_version: str,
    model: str,
    logid: str,
    max_output_tokens: int,
    max_chars: int,
) -> dict[str, str]:
    pdf_path: Path | None = None
    try:
        client = create_client(api_key, endpoint, api_version)
        pdf_path = download_pdf(task.pdf_url)
        markdown = extract_markdown(pdf_path)
        answer = answer_query(
            client=client,
            model=model,
            logid=logid,
            max_output_tokens=max_output_tokens,
            paper_url=task.source_url,
            paper_text=trim_content(markdown, max_chars),
            query=task.query,
        )
        return {
            "url": task.source_url,
            "query": task.query,
            "answer": answer,
        }
    except Exception as exc:
        return {
            "url": task.source_url,
            "query": task.query,
            "answer": f"ERROR: {exc}",
        }
    finally:
        if pdf_path and pdf_path.exists():
            pdf_path.unlink()


def render_text(results: list[dict[str, str]]) -> str:
    blocks = []
    for index, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] URL: {item['url']}",
                    f"Query: {item['query']}",
                    "Answer:",
                    item["answer"],
                ]
            )
        )
    return "\n\n".join(blocks)


def main() -> int:
    args = parse_args()
    tasks = build_tasks(args.task)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    results: list[dict[str, str]] = []
    max_workers = min(args.workers, len(tasks))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_task,
                task,
                args.api_key,
                args.endpoint,
                args.api_version,
                args.model,
                args.logid,
                args.max_output_tokens,
                args.max_chars,
            )
            for task in tasks
        ]
        for future in futures:
            results.append(future.result())

    if args.format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(render_text(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
