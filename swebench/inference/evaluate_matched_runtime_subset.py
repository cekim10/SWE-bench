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
    runtime_event_path_count = 0
    backend_call_path_count = 0
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
        if record.get("runtime_event_path"):
            runtime_event_path_count += 1
        if record.get("backend_call_path"):
            backend_call_path_count += 1
    real_backend_trace_count = sum(
        count for provider, count in provider_counts.items() if provider != "stub"
    )
    return {
        "record_count": len(records),
        "provider_counts": dict(sorted(provider_counts.items())),
        "valid_trace_count": valid_trace_count,
        "invalid_instance_ids": invalid_instance_ids,
        "missing_trace_path_count": missing_trace_path_count,
        "runtime_event_path_count": runtime_event_path_count,
        "backend_call_path_count": backend_call_path_count,
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


def build_runtime_event_metadata(
    records: Iterable[Mapping[str, object]],
) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for record in records:
        trace_path = record.get("trace_path")
        runtime_event_path = record.get("runtime_event_path")
        if not trace_path or not runtime_event_path:
            continue
        metadata[str(Path(str(trace_path)).resolve())] = str(
            Path(str(runtime_event_path)).resolve()
        )
    return metadata


def build_backend_call_metadata(
    records: Iterable[Mapping[str, object]],
) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for record in records:
        trace_path = record.get("trace_path")
        backend_call_path = record.get("backend_call_path")
        if not trace_path or not backend_call_path:
            continue
        metadata[str(Path(str(trace_path)).resolve())] = str(
            Path(str(backend_call_path)).resolve()
        )
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
        runtime_event_paths_by_trace_path=build_runtime_event_metadata(left_subset),
        backend_call_paths_by_trace_path=build_backend_call_metadata(left_subset),
        run_summary=filtered_run_summary(left_subset),
    )
    right_report = analyze_trace_paths(
        [record["trace_path"] for record in right_subset],
        trace_metadata_by_path=build_trace_metadata(right_subset),
        runtime_event_paths_by_trace_path=build_runtime_event_metadata(right_subset),
        backend_call_paths_by_trace_path=build_backend_call_metadata(right_subset),
        run_summary=filtered_run_summary(right_subset),
    )

    left_mismatch = left_report["aggregate"]["abstraction_mismatch"]
    right_mismatch = right_report["aggregate"]["abstraction_mismatch"]
    left_bridge = left_report["aggregate"]["oracle_abstraction_bridge"]
    right_bridge = right_report["aggregate"]["oracle_abstraction_bridge"]
    left_locality = left_report["aggregate"].get("reuse_locality")
    right_locality = right_report["aggregate"].get("reuse_locality")
    left_runtime = left_report["aggregate"].get("runtime_behavior")
    right_runtime = right_report["aggregate"].get("runtime_behavior")
    left_latency = left_report["aggregate"].get("backend_latency")
    right_latency = right_report["aggregate"].get("backend_latency")

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
    if left_locality is not None and right_locality is not None:
        comparison["left_minus_right"].update(
            {
                "avg_accesses_per_segment": left_locality["avg_accesses_per_segment"]
                - right_locality["avg_accesses_per_segment"],
                "avg_revisits_per_segment": left_locality["avg_revisits_per_segment"]
                - right_locality["avg_revisits_per_segment"],
                "revisited_segment_fraction": left_locality["revisited_segment_fraction"]
                - right_locality["revisited_segment_fraction"],
                "avg_prompt_revisit_distance": left_locality["avg_prompt_revisit_distance"]
                - right_locality["avg_prompt_revisit_distance"],
            }
        )
    if left_runtime is not None and right_runtime is not None:
        comparison["left_minus_right"].update(
            {
                "resident_hit_rate": left_runtime["resident_hit_rate"]
                - right_runtime["resident_hit_rate"],
                "token_weighted_reuse_rate": left_runtime["token_weighted_reuse_rate"]
                - right_runtime["token_weighted_reuse_rate"],
                "resident_hits": left_runtime["resident_hits"]
                - right_runtime["resident_hits"],
                "misses": left_runtime["misses"] - right_runtime["misses"],
                "materializations": left_runtime["materializations"]
                - right_runtime["materializations"],
                "rematerializations": left_runtime["rematerializations"]
                - right_runtime["rematerializations"],
                "reused_tokens": left_runtime["reused_tokens"]
                - right_runtime["reused_tokens"],
                "materialized_tokens": left_runtime["materialized_tokens"]
                - right_runtime["materialized_tokens"],
                "lifecycle_reclaims": left_runtime["lifecycle_reclaims"]
                - right_runtime["lifecycle_reclaims"],
                "policy_reclaims": left_runtime["policy_reclaims"]
                - right_runtime["policy_reclaims"],
            }
        )
    if left_latency is not None and right_latency is not None:
        comparison["left_minus_right"].update(
            {
                "avg_duration_ms": left_latency["avg_duration_ms"]
                - right_latency["avg_duration_ms"],
                "avg_backend_roundtrip_ms": left_latency["avg_backend_roundtrip_ms"]
                - right_latency["avg_backend_roundtrip_ms"],
                "avg_frontend_overhead_ms": left_latency["avg_frontend_overhead_ms"]
                - right_latency["avg_frontend_overhead_ms"],
                "duration_ms_per_1k_prompt_tokens": left_latency[
                    "duration_ms_per_1k_prompt_tokens"
                ]
                - right_latency["duration_ms_per_1k_prompt_tokens"],
                "total_prompt_tokens": left_latency["total_prompt_tokens"]
                - right_latency["total_prompt_tokens"],
                "total_prompt_payload_tokens_estimate": left_latency[
                    "total_prompt_payload_tokens_estimate"
                ]
                - right_latency["total_prompt_payload_tokens_estimate"],
                "total_duplicate_prompt_tokens_estimate": left_latency[
                    "total_duplicate_prompt_tokens_estimate"
                ]
                - right_latency["total_duplicate_prompt_tokens_estimate"],
            }
        )
        comparison["prompt_token_inflation"] = (
            right_latency["total_prompt_tokens"] / left_latency["total_prompt_tokens"]
            if left_latency["total_prompt_tokens"]
            else 0.0
        )

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
    left_locality = left_agg.get("reuse_locality")
    right_locality = right_agg.get("reuse_locality")
    left_runtime = left_agg.get("runtime_behavior")
    right_runtime = right_agg.get("runtime_behavior")
    left_latency = left_agg.get("backend_latency")
    right_latency = right_agg.get("backend_latency")

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

    if left_locality is not None and right_locality is not None:
        locality_rows = [
            ("Avg accesses / segment", left_locality["avg_accesses_per_segment"], right_locality["avg_accesses_per_segment"]),
            ("Avg revisits / segment", left_locality["avg_revisits_per_segment"], right_locality["avg_revisits_per_segment"]),
            ("Revisited segment fraction", left_locality["revisited_segment_fraction"], right_locality["revisited_segment_fraction"]),
            ("Avg prompt revisit distance", left_locality["avg_prompt_revisit_distance"], right_locality["avg_prompt_revisit_distance"]),
        ]
        lines.extend(
            [
                "",
                "## Reuse Locality",
                "",
                "| Metric | Left | Right | Left-Right |",
                "| - | -: | -: | -: |",
            ]
        )
        for metric, left_value, right_value in locality_rows:
            delta = left_value - right_value
            lines.append(
                f"| {metric} | {left_value:.4f} | {right_value:.4f} | {delta:.4f} |"
            )

    if left_runtime is not None and right_runtime is not None:
        runtime_rows = [
            ("Resident hit rate", left_runtime["resident_hit_rate"], right_runtime["resident_hit_rate"]),
            ("Token-weighted reuse rate", left_runtime["token_weighted_reuse_rate"], right_runtime["token_weighted_reuse_rate"]),
            ("Resident hits", left_runtime["resident_hits"], right_runtime["resident_hits"]),
            ("Misses", left_runtime["misses"], right_runtime["misses"]),
            ("Materializations", left_runtime["materializations"], right_runtime["materializations"]),
            ("Rematerializations", left_runtime["rematerializations"], right_runtime["rematerializations"]),
            ("Reused tokens", left_runtime["reused_tokens"], right_runtime["reused_tokens"]),
            ("Materialized tokens", left_runtime["materialized_tokens"], right_runtime["materialized_tokens"]),
            ("Lifecycle reclaims", left_runtime["lifecycle_reclaims"], right_runtime["lifecycle_reclaims"]),
            ("Policy reclaims", left_runtime["policy_reclaims"], right_runtime["policy_reclaims"]),
        ]
        lines.extend(
            [
                "",
                "## Practical Runtime",
                "",
                "| Metric | Left | Right | Left-Right |",
                "| - | -: | -: | -: |",
            ]
        )
        for metric, left_value, right_value in runtime_rows:
            delta = left_value - right_value
            if isinstance(left_value, float) or isinstance(right_value, float):
                lines.append(
                    f"| {metric} | {left_value:.4f} | {right_value:.4f} | {delta:.4f} |"
                )
            else:
                lines.append(f"| {metric} | {left_value} | {right_value} | {delta} |")

    if left_latency is not None and right_latency is not None:
        latency_rows = [
            ("Avg duration ms", left_latency["avg_duration_ms"], right_latency["avg_duration_ms"]),
            (
                "Avg backend round-trip ms",
                left_latency["avg_backend_roundtrip_ms"],
                right_latency["avg_backend_roundtrip_ms"],
            ),
            (
                "Avg frontend overhead ms",
                left_latency["avg_frontend_overhead_ms"],
                right_latency["avg_frontend_overhead_ms"],
            ),
            (
                "ms per 1k prompt tokens",
                left_latency["duration_ms_per_1k_prompt_tokens"],
                right_latency["duration_ms_per_1k_prompt_tokens"],
            ),
            ("Total prompt tokens", left_latency["total_prompt_tokens"], right_latency["total_prompt_tokens"]),
            (
                "Prompt payload tokens (est.)",
                left_latency["total_prompt_payload_tokens_estimate"],
                right_latency["total_prompt_payload_tokens_estimate"],
            ),
            (
                "Duplicate prompt tokens (est.)",
                left_latency["total_duplicate_prompt_tokens_estimate"],
                right_latency["total_duplicate_prompt_tokens_estimate"],
            ),
            (
                "Frontend cache hit rate",
                left_latency["frontend_cache_hit_rate"],
                right_latency["frontend_cache_hit_rate"],
            ),
        ]
        lines.extend(
            [
                "",
                "## Request Latency",
                "",
                "| Metric | Left | Right | Left-Right |",
                "| - | -: | -: | -: |",
            ]
        )
        for metric, left_value, right_value in latency_rows:
            delta = left_value - right_value
            if isinstance(left_value, float) or isinstance(right_value, float):
                lines.append(
                    f"| {metric} | {left_value:.4f} | {right_value:.4f} | {delta:.4f} |"
                )
            else:
                lines.append(f"| {metric} | {left_value} | {right_value} | {delta} |")
        inflation = report["comparison"].get("prompt_token_inflation")
        if inflation is not None:
            lines.extend(
                [
                    "",
                    f"- Prompt token inflation (right / left): {float(inflation):.4f}",
                ]
            )

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
