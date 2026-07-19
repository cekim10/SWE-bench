#!/usr/bin/env python3

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from swebench.inference.trace.agentic import (
    TracedAgentRunner,
    build_demo_instance,
    load_instances_from_path,
    make_backend,
)


def parse_args():
    parser = ArgumentParser(
        description="Run a lightweight multi-step agent loop with semantic trace emission."
    )
    parser.add_argument(
        "--instances_path",
        type=str,
        help="Path to a JSON or JSONL file containing instances with instance_id, problem_statement, and file_contents.",
    )
    parser.add_argument(
        "--demo_instance",
        action="store_true",
        help="Run against a built-in demo instance for emitter correctness checks.",
    )
    parser.add_argument(
        "--provider",
        choices=["stub", "openai", "anthropic", "ollama", "groq", "vllm"],
        default="stub",
        help="Model backend to use. `stub` requires no API access and is useful for trace sanity checks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for provider=openai, provider=anthropic, provider=ollama, or provider=groq.",
    )
    parser.add_argument(
        "--trace_dir",
        type=str,
        required=True,
        help="Directory where semantic trace JSONL files will be written.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to a JSONL file containing workflow outputs and trace validation results.",
    )
    parser.add_argument(
        "--tenant_id",
        type=str,
        default="local",
        help="Tenant identifier recorded in the semantic trace.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=2,
        help="Maximum planner/coder/tester iterations per instance.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=5,
        help="Maximum number of retrieved files to include per instance.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds for OpenAI-compatible backends.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum SDK retries for OpenAI-compatible backends.",
    )
    args = parser.parse_args()
    if not args.demo_instance and not args.instances_path:
        parser.error("Provide either --demo_instance or --instances_path.")
    return args


def main():
    args = parse_args()
    if args.demo_instance:
        instances = [build_demo_instance()]
    else:
        instances = load_instances_from_path(args.instances_path)

    backend = make_backend(
        args.provider,
        args.model,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    runner = TracedAgentRunner(
        backend=backend,
        trace_dir=args.trace_dir,
        tenant_id=args.tenant_id,
        max_iterations=args.max_iterations,
        max_files=args.max_files,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for instance in instances:
            try:
                result = runner.run_instance(instance)
            except Exception as exc:
                result = {
                    "instance_id": instance.instance_id,
                    "status": "ERROR",
                    "provider": args.provider,
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "trace_path": None,
                    "trace_validation": {
                        "is_valid": False,
                        "errors": [f"{exc.__class__.__name__}: {exc}"],
                        "summary": {},
                    },
                }
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
