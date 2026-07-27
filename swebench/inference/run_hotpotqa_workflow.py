#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from swebench.inference.trace.agentic import (
    HotpotQATracedAgentRunner,
    WorkflowInstance,
    build_hotpotqa_demo_instance,
    load_instances_from_path,
    make_backend,
)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "context"


def _hotpot_context_to_files(context: Sequence[object]) -> dict[str, str]:
    files: dict[str, str] = {}
    for item in context:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, sentences = item
        title_text = str(title)
        if isinstance(sentences, Sequence) and not isinstance(sentences, (str, bytes)):
            content = " ".join(str(sentence) for sentence in sentences)
        else:
            content = str(sentences)
        files[f"context/{_slugify(title_text)}.md"] = f"# {title_text}\n\n{content}\n"
    return files


def load_hotpotqa_instances_from_path(
    path: str | Path,
    *,
    limit: int | None = None,
) -> List[WorkflowInstance]:
    resolved_path = Path(path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        rows: Iterable[Mapping[str, object]] = [payload]
    else:
        rows = [row for row in payload if isinstance(row, Mapping)]

    instances: List[WorkflowInstance] = []
    for index, row in enumerate(rows):
        if limit is not None and len(instances) >= limit:
            break
        question = str(row.get("question", "")).strip()
        context = row.get("context", [])
        if not question or not isinstance(context, Sequence):
            continue
        file_contents = _hotpot_context_to_files(context)
        if not file_contents:
            continue
        instance_id = str(row.get("_id") or f"hotpotqa-{index + 1}")
        supporting_facts = row.get("supporting_facts", [])
        supporting_titles = []
        if isinstance(supporting_facts, Sequence) and not isinstance(
            supporting_facts, (str, bytes)
        ):
            for item in supporting_facts:
                if isinstance(item, (list, tuple)) and item:
                    supporting_titles.append(str(item[0]))
        readme_text = (
            "Answer the question using explicit multi-hop reasoning across the provided "
            "context documents. Prioritize supporting titles first: "
            + ", ".join(sorted(set(supporting_titles)))
            if supporting_titles
            else "Answer the question using explicit multi-hop reasoning across the provided context documents."
        )
        instances.append(
            WorkflowInstance(
                instance_id=instance_id,
                problem_statement=question,
                file_contents=file_contents,
                readmes={"README.md": readme_text},
                metadata={
                    "answer": row.get("answer"),
                    "question_type": row.get("type"),
                    "supporting_facts": supporting_facts,
                },
            )
        )
    return instances


def parse_args():
    parser = ArgumentParser(
        description="Run a HotpotQA-style traced multi-hop QA workload."
    )
    parser.add_argument(
        "--instances_path",
        type=str,
        help="Path to a JSON or JSONL file containing WorkflowInstance-compatible payloads.",
    )
    parser.add_argument(
        "--hotpot_examples_path",
        type=str,
        help="Path to official HotpotQA-style JSON examples with question/context fields.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit when reading --hotpot_examples_path.",
    )
    parser.add_argument(
        "--demo_instance",
        action="store_true",
        help="Run against the built-in HotpotQA-style demo instance.",
    )
    parser.add_argument(
        "--provider",
        choices=["stub", "openai", "anthropic", "ollama", "groq", "vllm", "continuum"],
        default="stub",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--trace_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tenant_id", type=str, default="local")
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--max_files", type=int, default=8)
    parser.add_argument(
        "--prompt_runtime_mode",
        choices=["monolithic", "segment_aware"],
        default="segment_aware",
    )
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--hotpotqa_path",
        type=str,
        help=(
            "Path to a hotpotqa/hotpot checkout. When omitted, the runner looks for "
            "'./.external/hotpot'."
        ),
    )
    args = parser.parse_args()
    provided_inputs = [
        bool(args.demo_instance),
        bool(args.instances_path),
        bool(args.hotpot_examples_path),
    ]
    if sum(provided_inputs) != 1:
        parser.error(
            "Provide exactly one of --demo_instance, --instances_path, or --hotpot_examples_path."
        )
    return args


def main():
    args = parse_args()
    if args.demo_instance:
        instances = [build_hotpotqa_demo_instance()]
    elif args.hotpot_examples_path:
        instances = load_hotpotqa_instances_from_path(
            args.hotpot_examples_path,
            limit=args.limit,
        )
    else:
        instances = load_instances_from_path(args.instances_path)

    backend = make_backend(
        args.provider,
        args.model,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    runner = HotpotQATracedAgentRunner(
        backend=backend,
        trace_dir=args.trace_dir,
        tenant_id=args.tenant_id,
        max_iterations=args.max_iterations,
        max_files=args.max_files,
        prompt_runtime_mode=args.prompt_runtime_mode,
        hotpotqa_path=args.hotpotqa_path,
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
                    "agent_family": "hotpotqa",
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
