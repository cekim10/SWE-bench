#!/usr/bin/env python3

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from swebench.inference.trace.analysis import (
    analyze_trace_paths,
    load_run_output_records,
    summarize_run_output,
)


def is_valid_trace_record(record: Mapping[str, object]) -> bool:
    validation = record.get("trace_validation") or {}
    return bool(record.get("trace_path")) and bool(validation.get("is_valid", False))


def matched_records_by_instance(
    left_records: Iterable[Mapping[str, object]],
    right_records: Iterable[Mapping[str, object]],
) -> tuple[Dict[str, Mapping[str, object]], Dict[str, Mapping[str, object]], List[str]]:
    left_valid = {
        str(record["instance_id"]): record
        for record in left_records
        if is_valid_trace_record(record)
    }
    right_valid = {
        str(record["instance_id"]): record
        for record in right_records
        if is_valid_trace_record(record)
    }
    matched_ids = sorted(set(left_valid) & set(right_valid))
    return left_valid, right_valid, matched_ids


def filtered_run_summary(records: List[Mapping[str, object]]) -> Dict[str, object]:
    provider_counts: Dict[str, int] = {}
    valid_trace_count = 0
    invalid_instance_ids = []
    missing_trace_path_count = 0
    for record in records:
        provider = str(record.get("provider", "unknown"))
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        validation = record.get("trace_validation") or {}
        if validation.get("is_valid", False) and record.get("trace_path"):
            valid_trace_count += 1
        else:
            invalid_instance_ids.append(str(record.get("instance_id", "unknown")))
        if not record.get("trace_path"):
            missing_trace_path_count += 1
    real_backend_trace_count = sum(
        count for provider, count in provider_counts.items() if provider != "stub"
    )
    return {
        "record_count": len(records),
        "provider_counts": dict(sorted(provider_counts.items())),
        "valid_trace_count": valid_trace_count,
        "invalid_instance_ids": invalid_instance_ids,
        "missing_trace_path_count": missing_trace_path_count,
        "real_backend_trace_count": real_backend_trace_count,
        "stub_trace_count": provider_counts.get("stub", 0),
    }


def build_trace_metadata(records: Iterable[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    metadata: Dict[str, Dict[str, object]] = {}
    for record in records:
        trace_path = record.get("trace_path")
        if not trace_path:
            continue
        metadata[str(Path(str(trace_path)).resolve())] = {
            "provider": record.get("provider", "unknown"),
            "instance_id": record.get("instance_id", "unknown"),
            "status": record.get("status", "unknown"),
            "prompt_runtime_mode": record.get("prompt_runtime_mode", "unknown"),
        }
    return metadata


def build_matched_report(
    *,
    left_run_output_path: str | Path,
    right_run_output_path: str | Path,
    left_label: str,
    right_label: str,
) -> Dict[str, object]:
    left_records = load_run_output_records(left_run_output_path)
    right_records = load_run_output_records(right_run_output_path)
    left_valid, right_valid, matched_ids = matched_records_by_instance(left_records, right_records)

    left_subset = [left_valid[instance_id] for instance_id in matched_ids]
    right_subset = [right_valid[instance_id] for instance_id in matched_ids]

    left_report = analyze_trace_paths(
        [record["trace_path"] for record in left_subset],
        trace_metadata_by_path=build_trace_metadata(left_subset),
        run_summary=filtered_run_summary(left_subset),
    )
    right_report = analyze_trace_paths(
        [record["trace_path"] for record in right_subset],
        trace_metadata_by_path=build_trace_metadata(right_subset),
        run_summary=filtered_run_summary(right_subset),
    )

    left_mismatch = left_report["aggregate"]["abstraction_mismatch"]
    right_mismatch = right_report["aggregate"]["abstraction_mismatch"]
    left_bridge = left_report["aggregate"]["oracle_abstraction_bridge"]
    right_bridge = right_report["aggregate"]["oracle_abstraction_bridge"]

    comparison = {
        "matched_trace_count": len(matched_ids),
        "left_minus_right": {
            "mixed_lifecycle_prompt_rate": left_mismatch["mixed_lifecycle_prompt_rate"]
            - right_mismatch["mixed_lifecycle_prompt_rate"],
            "pinned_live_fraction": left_mismatch["pinned_live_fraction"]
            - right_mismatch["pinned_live_fraction"],
            "fragmentation_loss": left_mismatch["fragmentation_loss"]
            - right_mismatch["fragmentation_loss"],
            "avg_lifetime_spread": left_mismatch["avg_lifetime_spread"]
            - right_mismatch["avg_lifetime_spread"],
            "oracle_peak_hbm_savings_fraction": left_bridge["oracle_peak_hbm_savings_fraction"]
            - right_bridge["oracle_peak_hbm_savings_fraction"],
            "oracle_service_cost_savings_fraction": left_bridge["oracle_service_cost_savings_fraction"]
            - right_bridge["oracle_service_cost_savings_fraction"],
        },
    }

    return {
        "left_label": left_label,
        "right_label": right_label,
        "matched_instance_ids": matched_ids,
        "left": left_report,
        "right": right_report,
        "comparison": comparison,
        "left_unmatched_valid_instance_ids": sorted(set(left_valid) - set(matched_ids)),
        "right_unmatched_valid_instance_ids": sorted(set(right_valid) - set(matched_ids)),
        "left_source_summary": summarize_run_output(left_run_output_path),
        "right_source_summary": summarize_run_output(right_run_output_path),
    }


def render_matched_markdown(report: Mapping[str, object]) -> str:
    left = report["left"]
    right = report["right"]
    left_agg = left["aggregate"]
    right_agg = right["aggregate"]
    left_mismatch = left_agg["abstraction_mismatch"]
    right_mismatch = right_agg["abstraction_mismatch"]
    left_bridge = left_agg["oracle_abstraction_bridge"]
    right_bridge = right_agg["oracle_abstraction_bridge"]

    rows = [
        (
            "Trace count",
            left["trace_count"],
            right["trace_count"],
        ),
        (
            "Mixed lifecycle rate",
            left_mismatch["mixed_lifecycle_prompt_rate"],
            right_mismatch["mixed_lifecycle_prompt_rate"],
        ),
        (
            "Pinned live fraction",
            left_mismatch["pinned_live_fraction"],
            right_mismatch["pinned_live_fraction"],
        ),
        (
            "Fragmentation loss",
            left_mismatch["fragmentation_loss"],
            right_mismatch["fragmentation_loss"],
        ),
        (
            "Avg lifetime spread",
            left_mismatch["avg_lifetime_spread"],
            right_mismatch["avg_lifetime_spread"],
        ),
        (
            "Monolithic peak HBM bytes",
            left_bridge["monolithic_peak_hbm_bytes"],
            right_bridge["monolithic_peak_hbm_bytes"],
        ),
        (
            "Ideal segment peak HBM bytes",
            left_bridge["ideal_segment_peak_hbm_bytes"],
            right_bridge["ideal_segment_peak_hbm_bytes"],
        ),
        (
            "Oracle peak-HBM savings",
            left_bridge["oracle_peak_hbm_savings_fraction"],
            right_bridge["oracle_peak_hbm_savings_fraction"],
        ),
        (
            "Monolithic service cost",
            left_bridge["monolithic_service_cost_units"],
            right_bridge["monolithic_service_cost_units"],
        ),
        (
            "Ideal segment service cost",
            left_bridge["ideal_segment_service_cost_units"],
            right_bridge["ideal_segment_service_cost_units"],
        ),
        (
            "Oracle service-cost savings",
            left_bridge["oracle_service_cost_savings_fraction"],
            right_bridge["oracle_service_cost_savings_fraction"],
        ),
    ]

    lines = [
        "# Matched Runtime Subset Comparison",
        "",
        f"- Left: {report['left_label']}",
        f"- Right: {report['right_label']}",
        f"- Matched instances: {len(report['matched_instance_ids'])}",
        "",
        "| Metric | Left | Right | Left-Right |",
        "| - | -: | -: | -: |",
    ]
    for metric, left_value, right_value in rows:
        delta = left_value - right_value
        if isinstance(left_value, float) or isinstance(right_value, float):
            lines.append(
                f"| {metric} | {left_value:.4f} | {right_value:.4f} | {delta:.4f} |"
            )
        else:
            lines.append(f"| {metric} | {left_value} | {right_value} | {delta} |")

    lines.extend(
        [
            "",
            "## Matched Instances",
            "",
            *(f"- {instance_id}" for instance_id in report["matched_instance_ids"]),
        ]
    )
    if report["left_unmatched_valid_instance_ids"]:
        lines.extend(
            [
                "",
                f"## Left-Only Valid Instances ({report['left_label']})",
                "",
                *(f"- {instance_id}" for instance_id in report["left_unmatched_valid_instance_ids"]),
            ]
        )
    if report["right_unmatched_valid_instance_ids"]:
        lines.extend(
            [
                "",
                f"## Right-Only Valid Instances ({report['right_label']})",
                "",
                *(f"- {instance_id}" for instance_id in report["right_unmatched_valid_instance_ids"]),
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = ArgumentParser(
        description="Compare two traced runtime runs on the matched subset of instances with valid traces in both runs."
    )
    parser.add_argument("--left_run_output_path", required=True, type=str)
    parser.add_argument("--right_run_output_path", required=True, type=str)
    parser.add_argument("--left_label", default="left", type=str)
    parser.add_argument("--right_label", default="right", type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--markdown_path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_matched_report(
        left_run_output_path=args.left_run_output_path,
        right_run_output_path=args.right_run_output_path,
        left_label=args.left_label,
        right_label=args.right_label,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.markdown_path:
        markdown_path = Path(args.markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_matched_markdown(report), encoding="utf-8")

    print(
        f"Matched {len(report['matched_instance_ids'])} instances -> {output_path}"
    )


if __name__ == "__main__":
    main()
