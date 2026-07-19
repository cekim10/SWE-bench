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

    semantic_name = "swebench.inference.trace.semantic_state"
    semantic_spec = importlib.util.spec_from_file_location(
        semantic_name, TRACE_DIR / "semantic_state.py"
    )
    if semantic_spec is None or semantic_spec.loader is None:
        raise RuntimeError("failed to load semantic_state.py")
    semantic_module = importlib.util.module_from_spec(semantic_spec)
    sys.modules[semantic_name] = semantic_module
    semantic_spec.loader.exec_module(semantic_module)

    agentic_name = "swebench.inference.trace.agentic"
    agentic_spec = importlib.util.spec_from_file_location(
        agentic_name, TRACE_DIR / "agentic.py"
    )
    if agentic_spec is None or agentic_spec.loader is None:
        raise RuntimeError("failed to load agentic.py")
    agentic_module = importlib.util.module_from_spec(agentic_spec)
    sys.modules[agentic_name] = agentic_module
    agentic_spec.loader.exec_module(agentic_module)
    return semantic_module, agentic_module


SEMANTIC_STATE, AGENTIC = load_trace_modules()


class TraceAgenticRunnerTests(unittest.TestCase):
    class CapturingBackend(AGENTIC.StubModelBackend):
        def __init__(self) -> None:
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            return super().complete(**kwargs)

    def test_stub_runner_emits_replan_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AGENTIC.TracedAgentRunner(
                backend=AGENTIC.StubModelBackend(),
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
            )
            result = runner.run_instance(AGENTIC.build_demo_instance())
            self.assertEqual(result["status"], "RESOLVED")
            self.assertTrue(result["trace_validation"]["is_valid"])

            trace_path = Path(result["trace_path"])
            events = AGENTIC.load_trace_events(trace_path)
            ops = [event["op"] for event in events]
            self.assertIn("SUPERSEDE", ops)
            self.assertIn("READ", ops)
            self.assertIn("RELEASE", ops)
            self.assertGreaterEqual(result["trace_validation"]["summary"]["read_prompt_count"], 3)

    def test_repeated_runs_overwrite_trace_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AGENTIC.TracedAgentRunner(
                backend=AGENTIC.StubModelBackend(),
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
            )
            first = runner.run_instance(AGENTIC.build_demo_instance())
            second = runner.run_instance(AGENTIC.build_demo_instance())

            self.assertTrue(first["trace_validation"]["is_valid"])
            self.assertTrue(second["trace_validation"]["is_valid"])

            events = AGENTIC.load_trace_events(second["trace_path"])
            self.assertEqual(events[0]["ts"], 0)
            self.assertEqual(events[-1]["ts"], len(events) - 1)

    def test_validate_trace_flags_missing_prompt_reads(self):
        report = AGENTIC.validate_trace_events(
            [
                {
                    "ts": 0,
                    "op": "CREATE",
                    "workflow_id": "w",
                    "tenant_id": "t",
                    "state_id": "plan_v1",
                    "logical_key": "planner/plan",
                    "state_type": "plan",
                    "size_bytes": 4,
                    "token_count": 1,
                    "producer": "planner",
                    "owner_scope": "workflow",
                    "recompute_cost": 1.0,
                }
            ]
        )
        self.assertFalse(report.is_valid)
        self.assertIn("trace contains no READ events", report.errors)

    def test_instance_loader_supports_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "instances.jsonl"
            payload = {
                "instance_id": "demo",
                "problem_statement": "Fix it.",
                "file_contents": {"foo.py": "print('x')\n"},
            }
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            instances = AGENTIC.load_instances_from_path(path)
            self.assertEqual(len(instances), 1)
            self.assertEqual(instances[0].instance_id, "demo")

    def test_langgraph_style_runner_emits_router_and_reviewer_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AGENTIC.LangGraphStyleTracedAgentRunner(
                backend=AGENTIC.StubModelBackend(),
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
            )
            result = runner.run_instance(AGENTIC.build_demo_instance())
            self.assertEqual(result["status"], "RESOLVED")
            self.assertEqual(result["agent_family"], "langgraph_style")
            self.assertTrue(result["trace_validation"]["is_valid"])

            events = AGENTIC.load_trace_events(result["trace_path"])
            producers = {event.get("producer") for event in events if "producer" in event}
            logical_keys = {event.get("logical_key") for event in events if "logical_key" in event}
            self.assertIn("router", producers)
            self.assertIn("reviewer", producers)
            self.assertIn("router/decision", logical_keys)
            self.assertIn("reviewer/notes", logical_keys)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "langgraph not installed in this interpreter")
    def test_real_langgraph_runner_emits_router_and_reviewer_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = self.CapturingBackend()
            runner = AGENTIC.LangGraphTracedAgentRunner(
                backend=backend,
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
            )
            result = runner.run_instance(AGENTIC.build_demo_instance())
            self.assertEqual(result["status"], "RESOLVED")
            self.assertEqual(result["agent_family"], "langgraph")
            self.assertEqual(result["prompt_runtime_mode"], "segment_aware")
            self.assertTrue(result["trace_validation"]["is_valid"])
            self.assertGreater(len(backend.calls), 0)
            self.assertTrue(
                all(call.get("prompt_mode") == "segment_aware" for call in backend.calls)
            )
            self.assertTrue(
                all(call.get("prompt_group") is not None for call in backend.calls)
            )
            self.assertTrue(
                all(call.get("segment_request") is not None for call in backend.calls)
            )
            self.assertTrue(
                all(call.get("materializer_snapshot") is not None for call in backend.calls)
            )

            events = AGENTIC.load_trace_events(result["trace_path"])
            producers = {event.get("producer") for event in events if "producer" in event}
            logical_keys = {event.get("logical_key") for event in events if "logical_key" in event}
            self.assertIn("router", producers)
            self.assertIn("reviewer", producers)
            self.assertIn("router/decision", logical_keys)
            self.assertIn("reviewer/notes", logical_keys)
