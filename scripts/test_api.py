#!/usr/bin/env python3
"""独立 teacher relay API smoke；不访问 ShopSimulator。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rollout.teacher_client import TeacherClient, TeacherClientError, load_env_file
from rollout.progress import ProgressLogger


def display(value):
    return value if value not in (None, "") else "N/A"


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 teacher relay 单轮请求")
    parser.add_argument("prompt")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--style", choices=("chat_completions", "responses"))
    args = parser.parse_args()
    values = load_env_file(str(PROJECT_ROOT / ".env.teacher"))
    config = {key: os.environ.get(key, values.get(key, "")) for key in (
        "TEACHER_API_URL", "TEACHER_API_KEY", "TEACHER_API_MODEL",
        "TEACHER_API_STYLE", "TEACHER_REASONING_EFFORT",
    )}
    style = args.style or config["TEACHER_API_STYLE"] or "chat_completions"
    effort = config["TEACHER_REASONING_EFFORT"] or "high"
    print(f"Model: {display(config['TEACHER_API_MODEL'])}")
    print(f"API style: {style}")
    print(f"Reasoning effort: {effort}")
    print()
    try:
        progress = ProgressLogger()
        with TeacherClient(
            api_url=config["TEACHER_API_URL"], api_key=config["TEACHER_API_KEY"],
            model=config["TEACHER_API_MODEL"], api_style=style, reasoning_effort=effort,
            read_timeout=args.timeout,
            on_retry=progress.retry,
        ) as client:
            response = client.generate([{"role": "user", "content": args.prompt}])
    except (TeacherClientError, ValueError) as exc:
        print(f"API smoke failed: {exc}", file=sys.stderr)
        if isinstance(exc, TeacherClientError):
            print(f"Failure class: {exc.kind}", file=sys.stderr)
            print(f"Retries: {exc.retries}", file=sys.stderr)
        return 2
    print("Response:")
    print(response.text)
    print()
    print(f"HTTP: {response.status_code}")
    print(f"Latency: {response.latency_s:.2f} s")
    print(f"Retries: {response.retries}")
    print(f"Input tokens: {display(response.input_tokens)}")
    print(f"Output tokens: {display(response.output_tokens)}")
    print(f"Request ID: {display(response.request_id)}")
    print(f"API returned model: {display(response.model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
