#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ec_llm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from swebench.inference.evaluate_matched_runtime_subset import (
    build_backend_call_metadata,
    build_runtime_event_metadata,
    build_trace_metadata,
    filtered_run_summary,
    is_valid_trace_record,
    load_run_output_records,
)
from swebench.inference.trace.analysis import analyze_trace_paths


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
ANALYSIS_DIR = LOGS_DIR / "analysis"
FIGURE_DIR = LOGS_DIR / "paper_figures"

VLLM_MONO_BASE = LOGS_DIR / "vllm-langgraph-mono.jsonl"
VLLM_MONO_RETRY = LOGS_DIR / "vllm-langgraph-mono-retry-13579.jsonl"
VLLM_MONO_MERGED = LOGS_DIR / "vllm-langgraph-mono-merged.jsonl"

VLLM_SEG_BASE = LOGS_DIR / "vllm-langgraph-seg.jsonl"
VLLM_SEG_RETRY = LOGS_DIR / "vllm-langgraph-seg-retry-12907.jsonl"
VLLM_SEG_MERGED = LOGS_DIR / "vllm-langgraph-seg-merged.jsonl"

CONTINUUM_MONO = LOGS_DIR / "continuum-langgraph-mono.jsonl"

REPORT_JSON = ANALYSIS_DIR / "langgraph-continuum-3way-8.json"
REPORT_MD = ANALYSIS_DIR / "langgraph-continuum-3way-8.md"
FIGURE_PNG = FIGURE_DIR / "fig10_langgraph_continuum_3way_8way.png"
FIGURE_PDF = FIGURE_DIR / "fig10_langgraph_continuum_3way_8way.pdf"


SERIES = [
    ("vllm_mono", "vLLM", "#696969", "//"),
    ("continuum_mono", "Continuum", "#FFB226", "\\\\"),
    ("vllm_seg", "vLLM + SegmentRuntime", "#486EE2", None),
]


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 1.2,
        }
    )


def finish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=12, width=1.2, length=4)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def load_records(path: str | Path) -> List[Dict[str, object]]:
    return [dict(record) for record in load_run_output_records(path)]


def merge_run_outputs(
    *,
    base_path: Path,
    retry_path: Path,
    output_path: Path,
) -> List[Dict[str, object]]:
    base_records = load_records(base_path)
    retry_records = load_records(retry_path) if retry_path.exists() else []
    retry_by_id = {str(record["instance_id"]): record for record in retry_records}

    merged: List[Dict[str, object]] = []
    seen_ids = set()
    for record in base_records:
        instance_id = str(record["instance_id"])
        merged.append(retry_by_id.get(instance_id, record))
        seen_ids.add(instance_id)
    for instance_id, record in retry_by_id.items():
        if instance_id not in seen_ids:
            merged.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record) + "\n")
    return merged


def valid_record_map(records: Iterable[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    return {
        str(record["instance_id"]): record
        for record in records
        if is_valid_trace_record(record)
    }


def analyze_subset(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return analyze_trace_paths(
        [record["trace_path"] for record in records],
        trace_metadata_by_path=build_trace_metadata(records),
        runtime_event_paths_by_trace_path=build_runtime_event_metadata(records),
        backend_call_paths_by_trace_path=build_backend_call_metadata(records),
        run_summary=filtered_run_summary(list(records)),
    )


def build_system_summary(report: Mapping[str, object]) -> Dict[str, float | int]:
    agg = report["aggregate"]
    mismatch = agg["abstraction_mismatch"]
    runtime = agg["runtime_behavior"]
    backend = agg["backend_latency"]
    return {
        "trace_count": int(report["trace_count"]),
        "mixed_lifecycle_prompt_rate": float(mismatch["mixed_lifecycle_prompt_rate"]),
        "avg_lifetime_spread": float(mismatch["avg_lifetime_spread"]),
        "fragmentation_loss": float(mismatch["fragmentation_loss"]),
        "pinned_live_fraction": float(mismatch["pinned_live_fraction"]),
        "resident_hit_rate": float(runtime["resident_hit_rate"]),
        "token_weighted_reuse_rate": float(runtime["token_weighted_reuse_rate"]),
        "resident_hits": int(runtime["resident_hits"]),
        "reused_tokens": int(runtime["reused_tokens"]),
        "materialized_tokens": int(runtime["materialized_tokens"]),
        "request_count": int(backend["request_count"]),
        "total_prompt_tokens": int(backend["total_prompt_tokens"]),
        "total_completion_tokens": int(backend.get("total_completion_tokens", 0)),
        "avg_backend_roundtrip_ms": float(backend["avg_backend_roundtrip_ms"]),
        "avg_duration_ms": float(backend["avg_duration_ms"]),
    }


def render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# LangGraph SWE-bench 3-Way Comparison",
        "",
        f"- Matched instances: {len(report['matched_instance_ids'])}",
        "- Systems: vLLM, Continuum, vLLM + SegmentRuntime",
        "",
        "## Matched Instances",
        "",
    ]
    lines.extend(f"- `{instance_id}`" for instance_id in report["matched_instance_ids"])
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| System | Trace Count | Resident Hit Rate | Token-Weighted Reuse | Prompt Tokens | Completion Tokens | Avg Backend ms | Avg Duration ms |",
            "| - | -: | -: | -: | -: | -: | -: | -: |",
        ]
    )
    for key, label, _, _ in SERIES:
        metrics = report["systems"][key]["summary"]
        lines.append(
            "| "
            f"{label} | "
            f"{metrics['trace_count']} | "
            f"{metrics['resident_hit_rate']:.4f} | "
            f"{metrics['token_weighted_reuse_rate']:.4f} | "
            f"{metrics['total_prompt_tokens']} | "
            f"{metrics['total_completion_tokens']} | "
            f"{metrics['avg_backend_roundtrip_ms']:.2f} | "
            f"{metrics['avg_duration_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Lifecycle / Mismatch",
            "",
            "| System | Mixed Lifecycle Rate | Avg Lifetime Spread | Fragmentation Loss | Pinned Live Fraction |",
            "| - | -: | -: | -: | -: |",
        ]
    )
    for key, label, _, _ in SERIES:
        metrics = report["systems"][key]["summary"]
        lines.append(
            "| "
            f"{label} | "
            f"{metrics['mixed_lifecycle_prompt_rate']:.4f} | "
            f"{metrics['avg_lifetime_spread']:.4f} | "
            f"{metrics['fragmentation_loss']:.4f} | "
            f"{metrics['pinned_live_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `vLLM` vs `vLLM + SegmentRuntime` is the strict runtime comparison.",
            "- `Continuum` is reported as an external modified-vLLM stateful serving baseline.",
            "- Results are computed on the matched subset that completed with valid traces in all three systems.",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_three_way_figure(report: Mapping[str, object]) -> None:
    configure_plot_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    axes = axes.flatten()
    x = np.arange(len(SERIES))
    labels = [label for _, label, _, _ in SERIES]

    metric_specs = [
        ("resident_hit_rate", "Resident Hit Rate", "Rate"),
        ("token_weighted_reuse_rate", "Token-Weighted Reuse", "Rate"),
        ("total_prompt_tokens", "Prompt Tokens", "Tokens"),
        ("avg_backend_roundtrip_ms", "Backend Roundtrip", "ms"),
    ]

    for ax, (metric_key, title, ylabel) in zip(axes, metric_specs):
        values = [report["systems"][key]["summary"][metric_key] for key, _, _, _ in SERIES]
        for idx, (series_key, label, color, hatch) in enumerate(SERIES):
            ax.bar(
                x[idx],
                values[idx],
                width=0.58,
                color=color,
                edgecolor="black",
                linewidth=1.2,
                hatch=hatch,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=14, pad=6)
        finish_axes(ax)

    fig.suptitle("LangGraph SWE-bench (8-way matched subset)", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    merge_run_outputs(
        base_path=VLLM_MONO_BASE,
        retry_path=VLLM_MONO_RETRY,
        output_path=VLLM_MONO_MERGED,
    )
    merge_run_outputs(
        base_path=VLLM_SEG_BASE,
        retry_path=VLLM_SEG_RETRY,
        output_path=VLLM_SEG_MERGED,
    )

    system_files = {
        "vllm_mono": VLLM_MONO_MERGED,
        "continuum_mono": CONTINUUM_MONO,
        "vllm_seg": VLLM_SEG_MERGED,
    }
    system_records = {key: load_records(path) for key, path in system_files.items()}
    valid_maps = {key: valid_record_map(records) for key, records in system_records.items()}
    matched_ids = sorted(set.intersection(*(set(records) for records in valid_maps.values())))

    systems: Dict[str, Dict[str, object]] = {}
    for key, _, _, _ in SERIES:
        subset = [valid_maps[key][instance_id] for instance_id in matched_ids]
        report = analyze_subset(subset)
        systems[key] = {
            "label": next(label for series_key, label, _, _ in SERIES if series_key == key),
            "run_output_path": str(system_files[key]),
            "report": report,
            "summary": build_system_summary(report),
        }

    output = {
        "matched_instance_ids": matched_ids,
        "matched_trace_count": len(matched_ids),
        "systems": systems,
        "notes": {
            "strict_runtime_comparison": "vLLM vs vLLM + SegmentRuntime",
            "external_stateful_baseline": "Continuum",
        },
    }

    REPORT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(output), encoding="utf-8")
    plot_three_way_figure(output)

    print(f"wrote {VLLM_MONO_MERGED}")
    print(f"wrote {VLLM_SEG_MERGED}")
    print(f"wrote {REPORT_JSON}")
    print(f"wrote {REPORT_MD}")
    print(f"wrote {FIGURE_PNG}")
    print(f"wrote {FIGURE_PDF}")


if __name__ == "__main__":
    main()
