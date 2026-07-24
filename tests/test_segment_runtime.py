import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "swebench" / "inference" / "runtime"


def load_runtime_module():
    for package_name in [
        "swebench",
        "swebench.inference",
        "swebench.inference.runtime",
    ]:
        if package_name not in sys.modules:
            module = types.ModuleType(package_name)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package_name] = module

    module_name = "swebench.inference.runtime.segment_materializer"
    spec = importlib.util.spec_from_file_location(
        module_name,
        RUNTIME_DIR / "segment_materializer.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load segment_materializer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


RUNTIME = load_runtime_module()


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class SegmentRuntimeTests(unittest.TestCase):
    def _segment(self, *, state_id: str, logical_key: str, role: str, text: str, version: int = 1):
        return RUNTIME.ContextSegment(
            state_id=state_id,
            identity=RUNTIME.SegmentIdentity(logical_key=logical_key, module=role),
            version=version,
            role=role,
            text=text,
            token_count=max(1, len(text) // 4),
            size_bytes=max(1, len(text.encode("utf-8"))),
        )

    def _lookup(self, runtime, segment):
        context = RUNTIME.ExecutionContextKey.from_lineage(
            segment_id=segment.segment_id,
            predecessor_segment_ids=segment.predecessor_ids,
            model_id="test-model",
            tokenizer_id="test-tokenizer",
            inference_config={"temperature": 0.0},
        )
        return runtime.lookup(segment, context)

    def test_release_defers_reclaim_for_evidence(self):
        clock = _Clock()
        runtime = RUNTIME.SegmentRuntime(
            clock=clock,
            reclaim_grace_by_role={"evidence": 5.0, "scratch": 0.0},
        )
        segment = self._segment(
            state_id="evidence_v1",
            logical_key="retrieval/foo.py",
            role="evidence",
            text="evidence body",
        )
        runtime.register(segment)
        self._lookup(runtime, segment)

        runtime.release(segment)
        record = runtime.get_record(segment.segment_id)
        self.assertEqual(record.semantic_state, RUNTIME.SemanticState.RELEASED)
        self.assertEqual(record.residency_state, RUNTIME.ResidencyState.RESIDENT)
        self.assertIsNotNone(record.pending_reclaim_at)

        clock.value += 4.0
        self.assertEqual(runtime.reclaim_expired(), 0)
        self.assertEqual(record.residency_state, RUNTIME.ResidencyState.RESIDENT)

        clock.value += 2.0
        self.assertEqual(runtime.reclaim_expired(), 1)
        self.assertEqual(record.residency_state, RUNTIME.ResidencyState.EVICTED)
        self.assertIsNone(record.pending_reclaim_at)

    def test_release_reclaims_scratch_immediately(self):
        clock = _Clock()
        runtime = RUNTIME.SegmentRuntime(
            clock=clock,
            reclaim_grace_by_role={"scratch": 0.0},
        )
        segment = self._segment(
            state_id="scratch_v1",
            logical_key="diagnostic/v1",
            role="scratch",
            text="temporary scratchpad",
        )
        runtime.register(segment)
        self._lookup(runtime, segment)

        runtime.release(segment)
        record = runtime.get_record(segment.segment_id)
        self.assertEqual(record.semantic_state, RUNTIME.SemanticState.RELEASED)
        self.assertEqual(record.residency_state, RUNTIME.ResidencyState.EVICTED)
        self.assertIsNone(record.pending_reclaim_at)

    def test_build_vllm_request_uses_fallback_prompt_and_selected_segments(self):
        runtime = RUNTIME.SegmentRuntime()
        task = self._segment(
            state_id="task_v1",
            logical_key="prompt/task",
            role="task",
            text="Fix bug #1",
        )
        evidence = self._segment(
            state_id="evidence_v1",
            logical_key="retrieval/foo.py",
            role="evidence",
            text="[start of foo.py]\nprint('x')\n[end of foo.py]",
        )
        runtime.register(task)
        runtime.register(evidence)
        group = runtime.build_group(
            group_id="planner-1",
            workflow_id="wf",
            consumer="planner",
            ordered_segment_ids=[task.segment_id, evidence.segment_id],
            request_segment_ids=[evidence.segment_id],
        )
        request = runtime.build_vllm_request(
            group,
            fallback_system_prompt="system prompt",
            fallback_user_prompt="Iteration: 1\nIssue:\nFix bug #1",
        )
        messages = request.to_openai_messages()
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "system prompt")
        self.assertIn("Iteration: 1", messages[1]["content"])
        self.assertIn("foo.py", messages[1]["content"])
        self.assertNotIn("Fix bug #1\n\nFix bug #1", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
