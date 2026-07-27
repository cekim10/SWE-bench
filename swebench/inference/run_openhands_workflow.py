#!/usr/bin/env python3

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from swebench.inference.trace.agentic import (
    OpenHandsTracedAgentRunner,
    build_demo_instance,
    load_instances_from_path,
    make_backend,
)


def parse_args():
    parser = ArgumentParser(
        description="Run an OpenHands prompt-backed traced multi-step workload."
    )
    parser.add_argument(
        "--instances_path",
        type=str,
        help="Path to a JSON or JSONL file containing instances with instance_id, problem_statement, and file_contents.",
    )
    parser.add_argument(
        "--demo_instance",
        action="store_true",
        help="Run against the built-in demo software-engineering instance.",
    )
    parser.add_argument(
        "--provider",
        choices=["stub", "openai", "anthropic", "ollama", "groq", "vllm"],
        default="stub",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--trace_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tenant_id", type=str, default="local")
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--max_files", type=int, default=5)
    parser.add_argument(
        "--prompt_runtime_mode",
        choices=["monolithic", "segment_aware"],
        default="segment_aware",
    )
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--openhands_path",
        type=str,
        help=(
            "Path to an OpenHands/openhands checkout. When omitted, the runner looks for "
            "'./.external/openhands'."
        ),
    )
    parser.add_argument(
        "--request_selection_profile",
        choices=[
            "default",
            "task_only",
            "task_latest_artifact",
            "task_plan",
            "task_plan_artifact",
            "task_plan_artifact_budgeted_evidence",
        ],
        default="default",
        help=(
            "OpenHands request-selection ablation profile for segment-aware runs."
        ),
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
    runner = OpenHandsTracedAgentRunner(
        backend=backend,
        trace_dir=args.trace_dir,
        tenant_id=args.tenant_id,
        max_iterations=args.max_iterations,
        max_files=args.max_files,
        prompt_runtime_mode=args.prompt_runtime_mode,
        openhands_path=args.openhands_path,
        request_selection_profile=args.request_selection_profile,
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
                    "agent_family": "openhands",
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
