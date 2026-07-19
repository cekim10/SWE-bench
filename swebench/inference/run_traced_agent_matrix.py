#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path

from swebench.inference.trace.agentic import (
    TracedAgentRunner,
    build_demo_instance,
    load_instances_from_path,
    make_backend,
)
from swebench.inference.trace.analysis import (
    analyze_trace_paths,
    discover_trace_paths,
    load_run_output_records,
    render_markdown_report,
    summarize_run_output,
)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "provider"


def parse_provider_spec(spec: str) -> tuple[str, str | None, str]:
    provider, separator, model = spec.partition(":")
    provider = provider.strip()
    model = model.strip() or None
    if not provider:
        raise ValueError(f"invalid provider spec {spec!r}")
    label = provider if model is None else f"{provider}-{sanitize_filename(model)}"
    return provider, model, label


def parse_args():
    parser = ArgumentParser(
        description="Run the traced multi-step agent against multiple providers and emit a combined comparison report."
    )
    parser.add_argument(
        "--instances_path",
        type=str,
        help="Path to a JSON or JSONL file containing traced-agent instances.",
    )
    parser.add_argument(
        "--demo_instance",
        action="store_true",
        help="Run against the built-in demo instance.",
    )
    parser.add_argument(
        "--provider_spec",
        action="append",
        required=True,
        help="Provider spec in the form provider[:model]. Repeat this flag for each backend, e.g. --provider_spec stub --provider_spec ollama:qwen2.5-coder --provider_spec groq:llama-3.3-70b-versatile",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for per-provider outputs, traces, combined JSONL, and analysis artifacts.",
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
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip the combined analyzer pass.",
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

    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    traces_dir = output_dir / "traces"
    runs_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    combined_output_path = output_dir / "combined.jsonl"
    parsed_specs = [parse_provider_spec(spec) for spec in args.provider_spec]

    with combined_output_path.open("w", encoding="utf-8") as combined_handle:
        for provider, model, label in parsed_specs:
            backend = make_backend(
                provider,
                model,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            provider_trace_dir = traces_dir / label
            provider_trace_dir.mkdir(parents=True, exist_ok=True)
            runner = TracedAgentRunner(
                backend=backend,
                trace_dir=provider_trace_dir,
                tenant_id=args.tenant_id,
                max_iterations=args.max_iterations,
                max_files=args.max_files,
            )

            provider_output_path = runs_dir / f"{label}.jsonl"
            with provider_output_path.open("w", encoding="utf-8") as provider_handle:
                for instance in instances:
                    try:
                        result = runner.run_instance(instance)
                    except Exception as exc:
                        result = {
                            "instance_id": instance.instance_id,
                            "status": "ERROR",
                            "provider": provider,
                            "error": f"{exc.__class__.__name__}: {exc}",
                            "trace_path": None,
                            "trace_validation": {
                                "is_valid": False,
                                "errors": [f"{exc.__class__.__name__}: {exc}"],
                                "summary": {},
                            },
                        }
                    provider_handle.write(json.dumps(result) + "\n")
                    combined_handle.write(json.dumps(result) + "\n")

    if args.skip_analysis:
        print(f"Completed {len(parsed_specs)} provider runs -> {combined_output_path}")
        return

    trace_paths = discover_trace_paths(run_output_path=combined_output_path)
    trace_metadata_by_path = {}
    for record in load_run_output_records(combined_output_path):
        trace_path = record.get("trace_path")
        if not trace_path:
            continue
        trace_metadata_by_path[str(Path(str(trace_path)).resolve())] = {
            "provider": record.get("provider", "unknown"),
            "instance_id": record.get("instance_id", "unknown"),
            "status": record.get("status", "unknown"),
        }
    report = analyze_trace_paths(
        trace_paths,
        trace_metadata_by_path=trace_metadata_by_path,
        run_summary=summarize_run_output(combined_output_path),
    )
    analysis_json_path = output_dir / "analysis.json"
    analysis_md_path = output_dir / "analysis.md"
    analysis_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    analysis_md_path.write_text(render_markdown_report(report), encoding="utf-8")
    print(
        f"Completed {len(parsed_specs)} provider runs -> {combined_output_path} "
        f"and {analysis_json_path}"
    )


if __name__ == "__main__":
    main()
