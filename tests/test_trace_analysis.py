import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "swebench" / "inference" / "trace"
RUNTIME_DIR = ROOT / "swebench" / "inference" / "runtime"


def load_trace_modules():
    for package_name in [
        "swebench",
        "swebench.inference",
        "swebench.inference.trace",
        "swebench.inference.runtime",
    ]:
        if package_name not in sys.modules:
            module = types.ModuleType(package_name)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package_name] = module

    for module_name, filename in [
        ("swebench.inference.runtime.segment_materializer", RUNTIME_DIR / "segment_materializer.py"),
        ("swebench.inference.runtime.vllm_adapter", RUNTIME_DIR / "vllm_adapter.py"),
    ]:
        spec = importlib.util.spec_from_file_location(module_name, filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    loaded = {}
    for module_name, filename in [
        ("swebench.inference.trace.semantic_state", "semantic_state.py"),
        ("swebench.inference.trace.agentic", "agentic.py"),
        ("swebench.inference.trace.analysis", "analysis.py"),
    ]:
        spec = importlib.util.spec_from_file_location(module_name, TRACE_DIR / filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded


MODULES = load_trace_modules()
AGENTIC = MODULES["swebench.inference.trace.agentic"]
ANALYSIS = MODULES["swebench.inference.trace.analysis"]


class TraceAnalysisTests(unittest.TestCase):
    def test_stub_trace_analysis_reports_modules_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_event_path = Path(tmpdir) / "runtime.jsonl"
            runner = AGENTIC.TracedAgentRunner(
                backend=AGENTIC.StubModelBackend(),
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
            )
            result = runner.run_instance(AGENTIC.build_demo_instance())
            runtime_event_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "workflow_id": result["instance_id"],
                                "segment_id": "system_v1",
                                "logical_id": "prompt/system",
                                "version": 1,
                                "role": "system",
                                "module": "system",
                                "operation": "REGISTER",
                                "timestamp": 0.0,
                                "from_semantic_state": None,
                                "to_semantic_state": "REGISTERED",
                                "from_residency_state": None,
                                "to_residency_state": "UNMATERIALIZED",
                                "lookup_status": None,
                                "execution_context_digest": None,
                                "reason": None,
                                "size_bytes": 10,
                            }
                        ),
                        json.dumps(
                            {
                                "workflow_id": result["instance_id"],
                                "segment_id": "system_v1",
                                "logical_id": "prompt/system",
                                "version": 1,
                                "role": "system",
                                "module": "system",
                                "operation": "MATERIALIZE_INTERNAL",
                                "timestamp": 1.0,
                                "from_semantic_state": "REGISTERED",
                                "to_semantic_state": "REGISTERED",
                                "from_residency_state": "UNMATERIALIZED",
                                "to_residency_state": "RESIDENT",
                                "lookup_status": "MISS",
                                "execution_context_digest": "ctx-1",
                                "reason": "first_materialization",
                                "size_bytes": 10,
                            }
                        ),
                        json.dumps(
                            {
                                "workflow_id": result["instance_id"],
                                "segment_id": "system_v1",
                                "logical_id": "prompt/system",
                                "version": 1,
                                "role": "system",
                                "module": "system",
                                "operation": "REUSE",
                                "timestamp": 2.0,
                                "from_semantic_state": "REGISTERED",
                                "to_semantic_state": "REGISTERED",
                                "from_residency_state": "RESIDENT",
                                "to_residency_state": "RESIDENT",
                                "lookup_status": "HIT_RESIDENT",
                                "execution_context_digest": "ctx-1",
                                "reason": "exact_context_match",
                                "size_bytes": 10,
                            }
                        ),
                        json.dumps(
                            {
                                "workflow_id": result["instance_id"],
                                "segment_id": "system_v1",
                                "logical_id": "prompt/system",
                                "version": 1,
                                "role": "system",
                                "module": "system",
                                "operation": "RECLAIM",
                                "timestamp": 3.0,
                                "from_semantic_state": "REGISTERED",
                                "to_semantic_state": "REGISTERED",
                                "from_residency_state": "RESIDENT",
                                "to_residency_state": "EVICTED",
                                "lookup_status": None,
                                "execution_context_digest": "ctx-1",
                                "reason": "release_reclamation",
                                "size_bytes": 10,
                            }
                        ),
                        json.dumps(
                            {
                                "workflow_id": result["instance_id"],
                                "segment_id": "system_v1",
                                "logical_id": "prompt/system",
                                "version": 1,
                                "role": "system",
                                "module": "system",
                                "operation": "RELEASE",
                                "timestamp": 4.0,
                                "from_semantic_state": "REGISTERED",
                                "to_semantic_state": "RELEASED",
                                "from_residency_state": "RESIDENT",
                                "to_residency_state": "EVICTED",
                                "lookup_status": None,
                                "execution_context_digest": None,
                                "reason": "semantic_release",
                                "size_bytes": 10,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = ANALYSIS.analyze_trace_paths(
                [result["trace_path"]],
                runtime_event_paths_by_trace_path={
                    str(Path(result["trace_path"]).resolve()): str(runtime_event_path.resolve())
                },
            )
            aggregate = report["aggregate"]

            modules = {
                row["module"]
                for row in aggregate["prompt_composition"]["module_rows"]
            }
            self.assertIn("system", modules)
            self.assertIn("task", modules)
            self.assertIn("plan", modules)
            self.assertIn("evidence", modules)
            self.assertGreater(
                aggregate["abstraction_mismatch"]["monolithic_invalidation_events"],
                0,
            )
            self.assertGreater(
                aggregate["abstraction_mismatch"]["mixed_lifecycle_prompt_rate"],
                0.0,
            )
            self.assertIn(
                "lifetime_spread_per_prompt",
                aggregate["abstraction_mismatch"],
            )
            self.assertGreater(
                len(aggregate["abstraction_mismatch"]["lifetime_spread_per_prompt"]),
                0,
            )
            self.assertGreaterEqual(
                aggregate["abstraction_mismatch"]["pinned_live_fraction"],
                0.0,
            )
            self.assertGreaterEqual(
                aggregate["abstraction_mismatch"]["fragmentation_loss"],
                0.0,
            )
            self.assertIn("lifetime_correlation", aggregate)
            self.assertIn("oracle_abstraction_bridge", aggregate)
            self.assertIn("abstraction_bridge", aggregate)
            self.assertGreater(
                aggregate["lifetime_correlation"]["prompt_vector_count"],
                0,
            )
            self.assertGreater(
                len(aggregate["lifetime_correlation"]["module_order"]),
                0,
            )
            self.assertEqual(
                len(aggregate["lifetime_correlation"]["correlation_matrix"]),
                len(aggregate["lifetime_correlation"]["module_order"]),
            )
            self.assertGreater(
                aggregate["oracle_abstraction_bridge"]["monolithic_pinned_live_bytes"],
                0,
            )
            self.assertEqual(
                aggregate["oracle_abstraction_bridge"]["ideal_segment_pinned_live_bytes"],
                0,
            )
            self.assertGreater(
                aggregate["oracle_abstraction_bridge"]["monolithic_peak_hbm_bytes"],
                aggregate["oracle_abstraction_bridge"]["ideal_segment_peak_hbm_bytes"],
            )
            self.assertGreater(
                aggregate["oracle_abstraction_bridge"]["monolithic_service_cost_units"],
                aggregate["oracle_abstraction_bridge"]["ideal_segment_service_cost_units"],
            )
            self.assertGreater(
                aggregate["oracle_abstraction_bridge"]["oracle_recompute_savings_fraction"],
                0.0,
            )
            self.assertGreater(
                aggregate["oracle_abstraction_bridge"]["oracle_rematerialization_savings_fraction"],
                0.0,
            )
            self.assertIn("runtime_behavior", aggregate)
            self.assertGreater(aggregate["runtime_behavior"]["event_count"], 0)
            self.assertGreater(aggregate["runtime_behavior"]["materializations"], 0)
            self.assertEqual(aggregate["runtime_behavior"]["policy_reclaims"], 0)
            self.assertGreater(
                len(aggregate["runtime_behavior"]["role_rows"]),
                0,
            )

    def test_markdown_render_contains_sections(self):
        report = {
            "trace_count": 1,
            "aggregate": {
                "state_count": 1,
                "prompt_call_count": 1,
                "prompt_composition": {
                    "module_rows": [
                        {
                            "module": "system",
                            "prompt_presence_rate": 1.0,
                            "avg_segments_per_prompt": 1.0,
                            "avg_bytes_per_prompt": 10.0,
                            "immutable_fraction": 1.0,
                            "shared_fraction": 1.0,
                            "ephemeral_fraction": 0.0,
                        }
                    ]
                },
                "lifecycle_characterization": {
                    "module_rows": [
                        {
                            "module": "system",
                            "count": 1,
                            "avg_lifetime": 10.0,
                            "median_lifetime": 10.0,
                            "avg_reads": 2.0,
                            "superseded_fraction": 0.0,
                            "released_fraction": 1.0,
                        }
                    ]
                },
                "lifetime_correlation": {
                    "module_order": ["system"],
                    "prompt_vector_count": 1,
                    "prompt_vectors": [{"system": 10.0}],
                    "correlation_matrix": [[1.0]],
                    "covariance_matrix": [[0.0]],
                    "heatmap_matrix": [["++"]],
                    "pair_count_matrix": [[1]],
                },
                "abstraction_mismatch": {
                    "evaluable_prompt_count": 1,
                    "monolithic_invalidation_events": 0,
                    "mixed_lifecycle_prompt_rate": 0.0,
                    "stale_bytes_before_next_prompt": 0,
                    "reusable_live_bytes": 0,
                    "total_live_bytes_in_evaluable_prompts": 10,
                    "pinned_live_fraction": 0.0,
                    "fragmentation_loss": 0.0,
                    "avg_lifetime_spread": 0.0,
                    "lifetime_spread_per_prompt": [0.0],
                },
                "oracle_abstraction_bridge": {
                    "transition_count": 1,
                    "monolithic_peak_hbm_bytes": 20,
                    "ideal_segment_peak_hbm_bytes": 10,
                    "monolithic_pinned_live_bytes": 10,
                    "ideal_segment_pinned_live_bytes": 0,
                    "monolithic_stale_retained_bytes": 5,
                    "ideal_segment_stale_retained_bytes": 0,
                    "monolithic_rematerialized_live_bytes": 10,
                    "ideal_segment_rematerialized_live_bytes": 0,
                    "monolithic_reload_bytes": 5,
                    "ideal_segment_reload_bytes": 0,
                    "monolithic_recompute_bytes": 10,
                    "ideal_segment_recompute_bytes": 0,
                    "monolithic_service_cost_units": 4.0,
                    "ideal_segment_service_cost_units": 0.0,
                    "oracle_peak_hbm_savings_fraction": 0.5,
                    "oracle_pinned_live_savings_fraction": 1.0,
                    "oracle_reload_savings_fraction": 1.0,
                    "oracle_recompute_savings_fraction": 1.0,
                    "oracle_rematerialization_savings_fraction": 1.0,
                    "oracle_service_cost_savings_fraction": 1.0,
                },
                "runtime_behavior": {
                    "event_count": 8,
                    "resident_hit_rate": 0.25,
                    "resident_hits": 1,
                    "evicted_hits": 0,
                    "misses": 3,
                    "invalid_lookups": 0,
                    "materializations": 3,
                    "rematerializations": 1,
                    "reuse_count": 1,
                    "reused_tokens": 12,
                    "materialized_tokens": 48,
                    "rematerialized_tokens": 16,
                    "token_weighted_reuse_rate": 0.2,
                    "lifecycle_reclaims": 2,
                    "policy_reclaims": 0,
                    "bytes_reclaimed": 20,
                    "role_rows": [
                        {
                            "role": "system",
                            "registrations": 1,
                            "resident_hits": 1,
                            "misses": 1,
                            "materializations": 1,
                            "reused_tokens": 12,
                            "materialized_tokens": 24,
                            "token_weighted_reuse_rate": 1 / 3,
                            "rematerializations": 0,
                            "lifecycle_reclaims": 1,
                            "policy_reclaims": 0,
                            "registered_but_never_reused": 0,
                            "reused_exactly_once": 1,
                            "reused_more_than_five": 0,
                            "resident_hit_rate": 0.5,
                        }
                    ],
                    "lifetime_rows": [
                        {
                            "role": "system",
                            "count": 1,
                            "avg_semantic_lifetime": 10.0,
                            "avg_registration_to_first_lookup": 1.0,
                            "avg_lookup_span": 4.0,
                            "registered_but_never_reused_ratio": 0.0,
                        }
                    ],
                },
                "abstraction_bridge": {
                    "transition_count": 1,
                    "monolithic_peak_hbm_bytes": 20,
                    "segment_aware_peak_hbm_bytes": 10,
                    "monolithic_pinned_live_bytes": 10,
                    "segment_aware_pinned_live_bytes": 0,
                    "monolithic_stale_retained_bytes": 5,
                    "segment_aware_stale_retained_bytes": 0,
                    "monolithic_rematerialized_live_bytes": 10,
                    "segment_aware_rematerialized_live_bytes": 0,
                    "monolithic_reload_bytes": 5,
                    "segment_aware_reload_bytes": 0,
                    "monolithic_recompute_bytes": 10,
                    "segment_aware_recompute_bytes": 0,
                    "monolithic_service_cost_units": 4.0,
                    "segment_aware_service_cost_units": 0.0,
                    "peak_hbm_savings_fraction": 0.5,
                    "pinned_live_savings_fraction": 1.0,
                    "reload_savings_fraction": 1.0,
                    "recompute_savings_fraction": 1.0,
                    "rematerialization_savings_fraction": 1.0,
                    "service_cost_savings_fraction": 1.0,
                },
            },
            "run_summary": {
                "record_count": 1,
                "provider_counts": {"stub": 1},
                "valid_trace_count": 1,
                "invalid_instance_ids": [],
                "missing_trace_path_count": 0,
                "runtime_event_path_count": 1,
                "real_backend_trace_count": 0,
                "stub_trace_count": 1,
            },
            "providers": {
                "stub": {
                    "trace_count": 1,
                    "aggregate": {
                        "abstraction_mismatch": {
                            "mixed_lifecycle_prompt_rate": 0.0,
                            "pinned_live_fraction": 0.0,
                            "fragmentation_loss": 0.0,
                        }
                    },
                }
            },
        }
        markdown = ANALYSIS.render_markdown_report(report)
        self.assertIn("## Run Summary", markdown)
        self.assertIn("## Prompt Composition", markdown)
        self.assertIn("## Lifecycle Characterization", markdown)
        self.assertIn("## Lifetime Correlation Matrix", markdown)
        self.assertIn("## Abstraction Mismatch", markdown)
        self.assertIn("## Oracle Abstraction Bridge", markdown)
        self.assertIn("## Practical Runtime", markdown)
        self.assertIn("offline oracle upper bound", markdown)
        self.assertIn("## Provider Breakdown", markdown)
        self.assertIn("Pinned live fraction", markdown)
        self.assertIn("Fragmentation loss", markdown)

    def test_summarize_run_output_counts_real_backend_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "run.jsonl"
            output_path.write_text(
                "\n".join(
                    [
                        '{"instance_id":"a","provider":"stub","trace_path":"/tmp/a.jsonl","runtime_event_path":"/tmp/a_runtime.jsonl","trace_validation":{"is_valid":true}}',
                        '{"instance_id":"b","provider":"openai","trace_path":"/tmp/b.jsonl","trace_validation":{"is_valid":false}}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary = ANALYSIS.summarize_run_output(output_path)
            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(summary["stub_trace_count"], 1)
            self.assertEqual(summary["real_backend_trace_count"], 1)
            self.assertEqual(summary["valid_trace_count"], 1)
            self.assertEqual(summary["invalid_instance_ids"], ["b"])
            self.assertEqual(summary["runtime_event_path_count"], 1)

    def test_discover_trace_paths_skips_missing_trace_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.write_text('{"ts":0,"op":"READ"}\n', encoding="utf-8")
            output_path = Path(tmpdir) / "run.jsonl"
            output_path.write_text(
                "\n".join(
                    [
                        json.dumps({"instance_id": "a", "trace_path": str(trace_path)}),
                        json.dumps({"instance_id": "b", "trace_path": None}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            discovered = ANALYSIS.discover_trace_paths(run_output_path=output_path)
            self.assertEqual(discovered, [trace_path.resolve()])
