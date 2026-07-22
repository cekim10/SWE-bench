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
INFERENCE_DIR = ROOT / "swebench" / "inference"


def load_modules():
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

    loaded = {}
    for module_name, filename in [
        ("swebench.inference.runtime.segment_materializer", RUNTIME_DIR / "segment_materializer.py"),
        ("swebench.inference.runtime.vllm_adapter", RUNTIME_DIR / "vllm_adapter.py"),
        ("swebench.inference.trace.semantic_state", TRACE_DIR / "semantic_state.py"),
        ("swebench.inference.trace.agentic", TRACE_DIR / "agentic.py"),
        ("swebench.inference.trace.analysis", TRACE_DIR / "analysis.py"),
        ("swebench.inference.evaluate_matched_runtime_subset", INFERENCE_DIR / "evaluate_matched_runtime_subset.py"),
    ]:
        spec = importlib.util.spec_from_file_location(module_name, filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded


MODULES = load_modules()
AGENTIC = MODULES["swebench.inference.trace.agentic"]
MATCHED = MODULES["swebench.inference.evaluate_matched_runtime_subset"]


class MatchedRuntimeSubsetTests(unittest.TestCase):
    def test_build_matched_report_intersects_valid_instance_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            trace_dir = tmpdir / "traces"
            runtime_event_a = tmpdir / "a_runtime.jsonl"
            runtime_event_b = tmpdir / "b_runtime.jsonl"
            runtime_event_c = tmpdir / "c_runtime.jsonl"
            runner = AGENTIC.TracedAgentRunner(
                backend=AGENTIC.StubModelBackend(),
                trace_dir=trace_dir,
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
            )

            def make_instance(instance_id: str):
                return AGENTIC.WorkflowInstance(
                    instance_id=instance_id,
                    problem_statement="Fix it.",
                    file_contents={"foo.py": "print('x')\n"},
                )

            a = runner.run_instance(make_instance("a"))
            b = runner.run_instance(make_instance("b"))
            c = runner.run_instance(make_instance("c"))

            for path, instance_id in (
                (runtime_event_a, "a"),
                (runtime_event_b, "b"),
                (runtime_event_c, "c"),
            ):
                path.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "workflow_id": instance_id,
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
                                    "workflow_id": instance_id,
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
                                    "execution_context_digest": f"{instance_id}-ctx",
                                    "reason": "first_materialization",
                                    "size_bytes": 10,
                                }
                            ),
                            json.dumps(
                                {
                                    "workflow_id": instance_id,
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
                                    "execution_context_digest": f"{instance_id}-ctx",
                                    "reason": "exact_context_match",
                                    "size_bytes": 10,
                                }
                            ),
                            json.dumps(
                                {
                                    "workflow_id": instance_id,
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
                                    "execution_context_digest": f"{instance_id}-ctx",
                                    "reason": "release_reclamation",
                                    "size_bytes": 10,
                                }
                            ),
                            json.dumps(
                                {
                                    "workflow_id": instance_id,
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

            left_path = tmpdir / "left.jsonl"
            left_rows = [
                {**a, "runtime_event_path": str(runtime_event_a)},
                {**b, "runtime_event_path": str(runtime_event_b)},
                {
                    "instance_id": "left-error",
                    "status": "ERROR",
                    "provider": "stub",
                    "trace_path": None,
                    "trace_validation": {"is_valid": False, "errors": ["boom"], "summary": {}},
                },
            ]
            left_path.write_text(
                "\n".join(json.dumps(row) for row in left_rows) + "\n",
                encoding="utf-8",
            )

            right_path = tmpdir / "right.jsonl"
            right_rows = [
                {**b, "runtime_event_path": str(runtime_event_b)},
                {**c, "runtime_event_path": str(runtime_event_c)},
            ]
            right_path.write_text(
                "\n".join(json.dumps(row) for row in right_rows) + "\n",
                encoding="utf-8",
            )

            report = MATCHED.build_matched_report(
                left_run_output_path=left_path,
                right_run_output_path=right_path,
                left_label="mono",
                right_label="seg",
            )

            self.assertEqual(report["matched_instance_ids"], ["b"])
            self.assertEqual(report["left"]["trace_count"], 1)
            self.assertEqual(report["right"]["trace_count"], 1)
            self.assertIn("a", report["left_unmatched_valid_instance_ids"])
            self.assertIn("c", report["right_unmatched_valid_instance_ids"])
            self.assertIn("runtime_behavior", report["left"]["aggregate"])
            self.assertGreater(
                report["left"]["aggregate"]["runtime_behavior"]["event_count"],
                0,
            )

    def test_render_matched_markdown_mentions_labels_and_subset(self):
        report = {
            "left_label": "mono",
            "right_label": "seg",
            "matched_instance_ids": ["demo-1"],
            "left_unmatched_valid_instance_ids": [],
            "right_unmatched_valid_instance_ids": [],
            "left": {
                "trace_count": 1,
                "aggregate": {
                    "abstraction_mismatch": {
                        "mixed_lifecycle_prompt_rate": 0.8,
                        "pinned_live_fraction": 0.6,
                        "fragmentation_loss": 0.7,
                        "avg_lifetime_spread": 10.0,
                    },
                    "oracle_abstraction_bridge": {
                        "monolithic_peak_hbm_bytes": 100,
                        "ideal_segment_peak_hbm_bytes": 50,
                        "oracle_peak_hbm_savings_fraction": 0.5,
                        "monolithic_service_cost_units": 20.0,
                        "ideal_segment_service_cost_units": 10.0,
                        "oracle_service_cost_savings_fraction": 0.5,
                    },
                    "runtime_behavior": {
                        "resident_hit_rate": 0.25,
                        "resident_hits": 1,
                        "misses": 3,
                        "materializations": 3,
                        "rematerializations": 1,
                        "lifecycle_reclaims": 2,
                        "policy_reclaims": 0,
                    },
                },
            },
            "right": {
                "trace_count": 1,
                "aggregate": {
                    "abstraction_mismatch": {
                        "mixed_lifecycle_prompt_rate": 0.7,
                        "pinned_live_fraction": 0.5,
                        "fragmentation_loss": 0.6,
                        "avg_lifetime_spread": 9.0,
                    },
                    "oracle_abstraction_bridge": {
                        "monolithic_peak_hbm_bytes": 120,
                        "ideal_segment_peak_hbm_bytes": 60,
                        "oracle_peak_hbm_savings_fraction": 0.5,
                        "monolithic_service_cost_units": 25.0,
                        "ideal_segment_service_cost_units": 14.0,
                        "oracle_service_cost_savings_fraction": 0.44,
                    },
                    "runtime_behavior": {
                        "resident_hit_rate": 0.5,
                        "resident_hits": 2,
                        "misses": 2,
                        "materializations": 2,
                        "rematerializations": 0,
                        "lifecycle_reclaims": 1,
                        "policy_reclaims": 0,
                    },
                },
            },
        }
        markdown = MATCHED.render_matched_markdown(report)
        self.assertIn("# Matched Runtime Subset Comparison", markdown)
        self.assertIn("## Practical Runtime", markdown)
        self.assertIn("mono", markdown)
        self.assertIn("seg", markdown)
        self.assertIn("demo-1", markdown)
