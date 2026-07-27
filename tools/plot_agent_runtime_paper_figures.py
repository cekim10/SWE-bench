#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ec_llm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from swebench.inference.evaluate_matched_runtime_subset import build_matched_report


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
OUT_DIR = LOGS_DIR / "paper_figures"

WORKLOADS = [
    {
        "key": "swe_bench",
        "label": "SWE-bench",
        "domain": "Coding benchmark",
        "left_run_output_path": LOGS_DIR / "vllm-langgraph-mono.jsonl",
        "right_run_output_path": LOGS_DIR / "vllm-langgraph-seg.jsonl",
    },
    {
        "key": "swe_agent",
        "label": "SWE-agent",
        "domain": "Coding agent",
        "left_run_output_path": LOGS_DIR / "swe-agent-mono.jsonl",
        "right_run_output_path": LOGS_DIR / "swe-agent-seg.jsonl",
    },
    {
        "key": "hotpotqa",
        "label": "HotpotQA",
        "domain": "Multi-hop QA",
        "left_run_output_path": LOGS_DIR / "hotpotqa-mono.jsonl",
        "right_run_output_path": LOGS_DIR / "hotpotqa-seg.jsonl",
    },
    {
        "key": "open_deep_research",
        "label": "Open Deep Research",
        "domain": "Research agent",
        "left_run_output_path": LOGS_DIR / "open-deep-research-mono.jsonl",
        "right_run_output_path": LOGS_DIR / "open-deep-research-seg.jsonl",
    },
]

BRIGHT_PALETTE = {
    "black": "#000000",
    "dim_gray": "#696969",
    "dark_gray": "#A9A9A9",
    "blue": "#486EE2",
    "orange": "#FFB226",
    "red": "#D94B4B",
    "white": "#FFFFFF",
}

HATCH_BY_SERIES = {
    "Monolithic": "//",
    "Segment-aware": None,
}

SERIES_STYLE = {
    "Monolithic": {
        "color": BRIGHT_PALETTE["dim_gray"],
        "edgecolor": "black",
        "hatch": HATCH_BY_SERIES["Monolithic"],
    },
    "Segment-aware": {
        "color": BRIGHT_PALETTE["blue"],
        "edgecolor": "black",
        "hatch": HATCH_BY_SERIES["Segment-aware"],
    },
}

FIG_WIDTH = 5.83
FIG6_SIZE = (FIG_WIDTH, 4.55)
FIG7_SIZE = (FIG_WIDTH, 2.55)
FIG8_SIZE = (FIG_WIDTH, 2.55)
FIG9_SIZE = (FIG_WIDTH, 2.55)
AXIS_LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13
TITLE_FONT_SIZE = 16
BAR_WIDTH = 0.34


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
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, width=1.2, length=4)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def legend_handles() -> List[plt.Rectangle]:
    return [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=SERIES_STYLE[name]["color"],
            edgecolor=SERIES_STYLE[name]["edgecolor"],
            linewidth=1.2,
            hatch=SERIES_STYLE[name]["hatch"],
        )
        for name in ("Monolithic", "Segment-aware")
    ]


def load_reports() -> Dict[str, Dict[str, object]]:
    reports: Dict[str, Dict[str, object]] = {}
    for workload in WORKLOADS:
        reports[workload["key"]] = build_matched_report(
            left_run_output_path=workload["left_run_output_path"],
            right_run_output_path=workload["right_run_output_path"],
            left_label="Monolithic",
            right_label="Segment-aware",
        )
    return reports


def workload_metric_rows(reports: Mapping[str, Mapping[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for workload in WORKLOADS:
        report = reports[workload["key"]]
        mono = report["left"]["aggregate"]
        seg = report["right"]["aggregate"]
        comparison = report["comparison"]
        mono_mismatch = mono["abstraction_mismatch"]
        seg_runtime = seg["runtime_behavior"]
        mono_runtime = mono["runtime_behavior"]
        mono_backend = mono["backend_latency"]
        seg_backend = seg["backend_latency"]
        rows.append(
            {
                "key": workload["key"],
                "label": workload["label"],
                "domain": workload["domain"],
                "matched_trace_count": int(comparison["matched_trace_count"]),
                "mixed_lifecycle_prompt_rate": float(
                    mono_mismatch["mixed_lifecycle_prompt_rate"]
                ),
                "avg_lifetime_spread": float(mono_mismatch["avg_lifetime_spread"]),
                "fragmentation_loss": float(mono_mismatch["fragmentation_loss"]),
                "pinned_live_fraction": float(mono_mismatch["pinned_live_fraction"]),
                "resident_hit_rate_mono": float(mono_runtime["resident_hit_rate"]),
                "resident_hit_rate_seg": float(seg_runtime["resident_hit_rate"]),
                "token_weighted_reuse_rate_mono": float(
                    mono_runtime["token_weighted_reuse_rate"]
                ),
                "token_weighted_reuse_rate_seg": float(
                    seg_runtime["token_weighted_reuse_rate"]
                ),
                "resident_hits_seg": int(seg_runtime["resident_hits"]),
                "reused_tokens_seg": int(seg_runtime["reused_tokens"]),
                "requests_mono": int(mono_backend["request_count"]),
                "iterations_mono": int(mono_backend.get("iteration_count", 0)),
                "avg_prompt_tokens_mono": float(mono_backend["avg_prompt_tokens"]),
                "prompt_tokens_mono": int(mono_backend["total_prompt_tokens"]),
                "prompt_tokens_seg": int(seg_backend["total_prompt_tokens"]),
                "completion_tokens_mono": int(mono_backend["total_completion_tokens"]),
                "completion_tokens_seg": int(seg_backend["total_completion_tokens"]),
                "backend_latency_mono": float(mono_backend["avg_backend_roundtrip_ms"]),
                "backend_latency_seg": float(seg_backend["avg_backend_roundtrip_ms"]),
                "duration_mono": float(mono_backend["avg_duration_ms"]),
                "duration_seg": float(seg_backend["avg_duration_ms"]),
                "prompt_token_inflation": float(comparison["prompt_token_inflation"]),
            }
        )
    return rows


def draw_single_series_subplot(
    ax: plt.Axes,
    labels: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    color: str,
) -> None:
    x = np.arange(len(labels))
    ax.bar(
        x,
        values,
        width=0.58,
        color=color,
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=6)
    finish_axes(ax)


def draw_grouped_subplot(
    ax: plt.Axes,
    labels: List[str],
    mono_values: List[float],
    seg_values: List[float],
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(labels))
    offsets = np.array([-BAR_WIDTH / 2.0, BAR_WIDTH / 2.0])
    for idx, (name, values) in enumerate(
        [("Monolithic", mono_values), ("Segment-aware", seg_values)]
    ):
        style = SERIES_STYLE[name]
        ax.bar(
            x + offsets[idx],
            values,
            width=BAR_WIDTH,
            color=style["color"],
            edgecolor=style["edgecolor"],
            linewidth=1.2,
            hatch=style["hatch"],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=6)
    finish_axes(ax)


def draw_fig6(rows: List[Mapping[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=FIG6_SIZE)
    axes = axes.flatten()
    metrics = [
        ("mixed_lifecycle_prompt_rate", "Rate", "(a) Mixed Lifecycle Prompt Rate"),
        ("avg_lifetime_spread", "Spread", "(b) Average Lifetime Spread"),
        ("fragmentation_loss", "Loss", "(c) Fragmentation Loss"),
        ("pinned_live_fraction", "Fraction", "(d) Pinned Live Fraction"),
    ]
    for ax, (key, ylabel, title) in zip(axes, metrics):
        values = [float(row[key]) for row in rows]
        draw_single_series_subplot(
            ax,
            labels,
            values,
            ylabel=ylabel,
            title=title,
            color=BRIGHT_PALETTE["blue"],
        )
        if "rate" in key or "fraction" in key or "loss" in key:
            ax.set_ylim(0.0, max(1.0, max(values) * 1.12))
    fig.tight_layout()
    save_figure(fig, "fig6_semantic_context_lifecycle_characterization")


def draw_fig7(rows: List[Mapping[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=FIG7_SIZE)
    draw_grouped_subplot(
        axes[0],
        labels,
        [float(row["resident_hit_rate_mono"]) for row in rows],
        [float(row["resident_hit_rate_seg"]) for row in rows],
        ylabel="Rate",
        title="(a) Resident Hit Rate",
    )
    draw_grouped_subplot(
        axes[1],
        labels,
        [float(row["token_weighted_reuse_rate_mono"]) for row in rows],
        [float(row["token_weighted_reuse_rate_seg"]) for row in rows],
        ylabel="Rate",
        title="(b) Token-weighted Reuse Rate",
    )
    axes[1].legend(
        legend_handles(),
        ["Monolithic", "Segment-aware"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.27),
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.1,
        columnspacing=1.0,
    )
    fig.tight_layout()
    save_figure(fig, "fig7_semantic_reuse_recovery")


def draw_fig8(rows: List[Mapping[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=FIG8_SIZE)
    draw_grouped_subplot(
        axes[0],
        labels,
        [float(row["prompt_tokens_mono"]) for row in rows],
        [float(row["prompt_tokens_seg"]) for row in rows],
        ylabel="Tokens",
        title="(a) Prompt Tokens",
    )
    draw_grouped_subplot(
        axes[1],
        labels,
        [float(row["backend_latency_mono"]) for row in rows],
        [float(row["backend_latency_seg"]) for row in rows],
        ylabel="ms",
        title="(b) Backend Roundtrip Latency",
    )
    axes[1].legend(
        legend_handles(),
        ["Monolithic", "Segment-aware"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.27),
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.1,
        columnspacing=1.0,
    )
    fig.tight_layout()
    save_figure(fig, "fig8_prompt_construction_overhead")


def draw_fig9(rows: List[Mapping[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=FIG9_SIZE)
    draw_grouped_subplot(
        axes[0],
        labels,
        [float(row["completion_tokens_mono"]) for row in rows],
        [float(row["completion_tokens_seg"]) for row in rows],
        ylabel="Tokens",
        title="(a) Completion Tokens",
    )
    draw_grouped_subplot(
        axes[1],
        labels,
        [float(row["duration_mono"]) for row in rows],
        [float(row["duration_seg"]) for row in rows],
        ylabel="ms",
        title="(b) End-to-end Duration",
    )
    axes[1].legend(
        legend_handles(),
        ["Monolithic", "Segment-aware"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.27),
        ncol=2,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.1,
        columnspacing=1.0,
    )
    fig.tight_layout()
    save_figure(fig, "fig9_end_to_end_performance")


def render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, headers: List[str], rows: List[List[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_tables(rows: List[Mapping[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table2_headers = ["Workload", "Domain", "Requests", "Iterations", "Avg Prompt Tokens"]
    table2_rows = [
        [
            str(row["label"]),
            str(row["domain"]),
            str(int(row["requests_mono"])),
            str(int(row["iterations_mono"])),
            f"{float(row['avg_prompt_tokens_mono']):.1f}",
        ]
        for row in rows
    ]
    table2_md = render_markdown_table(table2_headers, table2_rows)
    (OUT_DIR / "table2_workload_summary.md").write_text(table2_md, encoding="utf-8")
    write_csv(OUT_DIR / "table2_workload_summary.csv", table2_headers, table2_rows)

    table3_headers = [
        "Workload",
        "Resident Hits",
        "Reused Tokens",
        "Prompt Inflation",
        "Avg Backend (mono->seg ms)",
        "Avg Duration (mono->seg ms)",
    ]
    table3_rows = [
        [
            str(row["label"]),
            str(int(row["resident_hits_seg"])),
            str(int(row["reused_tokens_seg"])),
            f"{float(row['prompt_token_inflation']):.3f}x",
            f"{float(row['backend_latency_mono']):.1f} -> {float(row['backend_latency_seg']):.1f}",
            f"{float(row['duration_mono']):.1f} -> {float(row['duration_seg']):.1f}",
        ]
        for row in rows
    ]
    table3_md = render_markdown_table(table3_headers, table3_rows)
    (OUT_DIR / "table3_detailed_runtime_statistics.md").write_text(
        table3_md,
        encoding="utf-8",
    )
    write_csv(
        OUT_DIR / "table3_detailed_runtime_statistics.csv",
        table3_headers,
        table3_rows,
    )


def write_summary_json(rows: List[Mapping[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in rows:
        serializable.append(dict(row))
    (OUT_DIR / "figure_table_metrics.json").write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    configure_plot_style()
    reports = load_reports()
    rows = workload_metric_rows(reports)
    write_summary_json(rows)
    draw_fig6(rows)
    draw_fig7(rows)
    draw_fig8(rows)
    draw_fig9(rows)
    write_tables(rows)
    print(f"Wrote figures and tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
