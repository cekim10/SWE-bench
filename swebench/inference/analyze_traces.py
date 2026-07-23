#!/usr/bin/env python3

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from swebench.inference.trace.analysis import (
    analyze_trace_paths,
    discover_trace_paths,
    load_run_output_records,
    render_markdown_report,
    summarize_run_output,
)


def parse_args():
    parser = ArgumentParser(
        description="Analyze semantic state traces for prompt composition, lifecycle, and abstraction mismatch."
    )
    parser.add_argument("--trace_dir", type=str, help="Directory containing trace JSONL files.")
    parser.add_argument("--run_output_path", type=str, help="JSONL output from run_traced_api_agent; trace paths will be extracted.")
    parser.add_argument("--trace_paths", nargs="*", help="Explicit trace JSONL paths.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to write JSON analysis output.")
    parser.add_argument("--markdown_path", type=str, help="Optional path to write a Markdown summary report.")
    args = parser.parse_args()
    if not args.trace_dir and not args.run_output_path and not args.trace_paths:
        parser.error("Provide at least one of --trace_dir, --run_output_path, or --trace_paths.")
    return args


def main():
    args = parse_args()
    trace_paths = discover_trace_paths(
        trace_dir=args.trace_dir,
        trace_paths=args.trace_paths,
        run_output_path=args.run_output_path,
    )
    trace_metadata_by_path = None
    runtime_event_paths_by_trace_path = None
    backend_call_paths_by_trace_path = None
    run_summary = None
    if args.run_output_path:
        run_summary = summarize_run_output(args.run_output_path)
        trace_metadata_by_path = {}
        runtime_event_paths_by_trace_path = {}
        backend_call_paths_by_trace_path = {}
        for record in load_run_output_records(args.run_output_path):
            trace_path = record.get("trace_path")
            if not trace_path:
                continue
            resolved_trace_path = str(Path(str(trace_path)).resolve())
            trace_metadata_by_path[resolved_trace_path] = {
                "provider": record.get("provider", "unknown"),
                "instance_id": record.get("instance_id", "unknown"),
                "status": record.get("status", "unknown"),
            }
            runtime_event_path = record.get("runtime_event_path")
            if runtime_event_path:
                runtime_event_paths_by_trace_path[resolved_trace_path] = str(
                    Path(str(runtime_event_path)).resolve()
                )
            backend_call_path = record.get("backend_call_path")
            if backend_call_path:
                backend_call_paths_by_trace_path[resolved_trace_path] = str(
                    Path(str(backend_call_path)).resolve()
                )
    report = analyze_trace_paths(
        trace_paths,
        trace_metadata_by_path=trace_metadata_by_path,
        runtime_event_paths_by_trace_path=runtime_event_paths_by_trace_path,
        backend_call_paths_by_trace_path=backend_call_paths_by_trace_path,
        run_summary=run_summary,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.markdown_path:
        markdown_path = Path(args.markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"Analyzed {len(trace_paths)} traces -> {output_path}")


if __name__ == "__main__":
    main()
