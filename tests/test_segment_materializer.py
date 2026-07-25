import dataclasses
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "swebench" / "inference" / "runtime"


def load_runtime_module():
    for package_name in ["swebench", "swebench.inference", "swebench.inference.runtime"]:
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
ContextSegment = RUNTIME.ContextSegment
ContextSegmentState = RUNTIME.ContextSegmentState
ExecutionContextKey = RUNTIME.ExecutionContextKey
LookupStatus = RUNTIME.LookupStatus
ReclaimNotSupportedError = RUNTIME.ReclaimNotSupportedError
ResidencyTier = RUNTIME.ResidencyTier
SegmentIdentity = RUNTIME.SegmentIdentity
SegmentMaterializer = RUNTIME.SegmentMaterializer
SegmentRole = RUNTIME.SegmentRole
SegmentVersion = RUNTIME.SegmentVersion
SemanticState = RUNTIME.SemanticState
SyntheticMaterializationAdapter = RUNTIME.SyntheticMaterializationAdapter
WeakHeuristicSegmentPolicy = RUNTIME.WeakHeuristicSegmentPolicy


class SegmentMaterializerTests(unittest.TestCase):
    def make_segment(
        self,
        *,
        state_id: str,
        logical_key: str,
        module: str,
        version: int,
        text: str,
        role: SegmentRole | str,
        is_ephemeral: bool = False,
        is_shared: bool = False,
        is_immutable: bool = False,
    ):
        return SegmentVersion(
            state_id=state_id,
            identity=SegmentIdentity(logical_key=logical_key, module=module),
            version=version,
            role=role,
            text=text,
            size_bytes=len(text.encode("utf-8")),
            token_count=max(1, len(text)),
            is_ephemeral=is_ephemeral,
            is_shared=is_shared,
            is_immutable=is_immutable,
            metadata={"role": role.value if isinstance(role, SegmentRole) else role},
        )

    def make_context(self, *, segment_id: str, predecessors=(), model_id="model-a", cfg=None):
        return ExecutionContextKey.from_lineage(
            segment_id=segment_id,
            predecessor_segment_ids=predecessors,
            model_id=model_id,
            tokenizer_id="tok-a",
            inference_config=cfg or {"temperature": 0.0},
        )

    def test_context_segment_is_frozen_and_tokenized_deterministically(self):
        segment = self.make_segment(
            state_id="system-1",
            logical_key="prompt/system",
            module="system",
            version=1,
            text="You are a planner.",
            role=SegmentRole.SYSTEM,
            is_shared=True,
            is_immutable=True,
        )

        self.assertEqual(segment.segment_id, "system-1")
        self.assertEqual(segment.logical_id, "prompt/system")
        self.assertEqual(segment.role, SegmentRole.SYSTEM)
        self.assertEqual(segment.content, tuple("You are a planner.".encode("utf-8")))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            segment.text = "mutated"

    def test_lookup_miss_then_hit_resident(self):
        adapter = SyntheticMaterializationAdapter()
        materializer = SegmentMaterializer(serving_adapter=adapter)
        plan = self.make_segment(
            state_id="plan-1",
            logical_key="planner/plan",
            module="plan",
            version=1,
            text="Plan v1",
            role=SegmentRole.PLAN,
        )
        materializer.register(plan)
        context = self.make_context(segment_id="plan-1")

        first = materializer.lookup(plan, context)
        self.assertEqual(first.status, LookupStatus.MISS)
        self.assertIsNotNone(first.association)
        self.assertEqual(materializer.get_record("plan-1").semantic_state, SemanticState.REGISTERED)
        self.assertEqual(len(adapter.materialize_calls), 1)

        second = materializer.lookup(plan, context)
        self.assertEqual(second.status, LookupStatus.HIT_RESIDENT)
        self.assertEqual(len(adapter.materialize_calls), 1)
        self.assertEqual(materializer.state_for("plan-1"), ContextSegmentState.RESIDENT)

    def test_lookup_after_eviction_returns_hit_evicted_and_rematerializes(self):
        adapter = SyntheticMaterializationAdapter()
        materializer = SegmentMaterializer(serving_adapter=adapter)
        evidence = self.make_segment(
            state_id="evidence-1",
            logical_key="retrieval/doc-1",
            module="evidence",
            version=1,
            text="Retrieved evidence",
            role=SegmentRole.EVIDENCE,
        )
        materializer.register(evidence)
        context = self.make_context(segment_id="evidence-1")

        first = materializer.lookup(evidence, context)
        self.assertEqual(first.status, LookupStatus.MISS)
        materializer.evict_segment("evidence-1")
        self.assertEqual(materializer.state_for("evidence-1"), ContextSegmentState.EVICTED)

        second = materializer.lookup(evidence, context)
        self.assertEqual(second.status, LookupStatus.HIT_EVICTED)
        self.assertEqual(len(adapter.materialize_calls), 2)
        self.assertEqual(materializer.state_for("evidence-1"), ContextSegmentState.RESIDENT)

    def test_different_execution_contexts_do_not_reuse(self):
        adapter = SyntheticMaterializationAdapter()
        materializer = SegmentMaterializer(serving_adapter=adapter)
        scratch = self.make_segment(
            state_id="scratch-1",
            logical_key="scratch/notes",
            module="scratchpad",
            version=1,
            text="temp",
            role=SegmentRole.SCRATCH,
            is_ephemeral=True,
        )
        materializer.register(scratch)
        a = self.make_context(segment_id="scratch-1", predecessors=("system-1",))
        b = self.make_context(segment_id="scratch-1", predecessors=("system-1", "plan-1"))

        first = materializer.lookup(scratch, a)
        second = materializer.lookup(scratch, b)

        self.assertEqual(first.status, LookupStatus.MISS)
        self.assertEqual(second.status, LookupStatus.MISS)
        self.assertEqual(len(adapter.materialize_calls), 2)
        self.assertEqual(len(materializer.get_record("scratch-1").associations), 2)

    def test_superseded_and_released_segments_become_invalid(self):
        adapter = SyntheticMaterializationAdapter()
        materializer = SegmentMaterializer(serving_adapter=adapter)
        old_plan = self.make_segment(
            state_id="plan-1",
            logical_key="planner/plan",
            module="plan",
            version=1,
            text="Plan v1",
            role=SegmentRole.PLAN,
        )
        new_plan = self.make_segment(
            state_id="plan-2",
            logical_key="planner/plan",
            module="plan",
            version=2,
            text="Plan v2",
            role=SegmentRole.PLAN,
        )
        materializer.register(old_plan)
        context = self.make_context(segment_id="plan-1")
        materializer.lookup(old_plan, context)
        materializer.supersede("plan-1", new_plan)

        invalid_old = materializer.lookup(old_plan, context)
        self.assertEqual(invalid_old.status, LookupStatus.INVALID)
        self.assertEqual(materializer.state_for("plan-1"), ContextSegmentState.INVALIDATED)
        self.assertEqual(materializer.current_segment_id_for("planner/plan"), "plan-2")

        materializer.release("plan-2")
        invalid_new = materializer.lookup(
            new_plan,
            self.make_context(segment_id="plan-2"),
        )
        self.assertEqual(invalid_new.status, LookupStatus.INVALID)
        self.assertEqual(materializer.state_for("plan-2"), ContextSegmentState.RELEASED)

    def test_prepare_group_read_promotes_and_assembles_segments(self):
        adapter = SyntheticMaterializationAdapter()
        materializer = SegmentMaterializer(serving_adapter=adapter)
        system = self.make_segment(
            state_id="system-1",
            logical_key="prompt/system",
            module="system",
            version=1,
            text="You are a planner.",
            role=SegmentRole.SYSTEM,
            is_shared=True,
            is_immutable=True,
        )
        plan = self.make_segment(
            state_id="plan-1",
            logical_key="planner/plan",
            module="plan",
            version=1,
            text="Plan v1",
            role=SegmentRole.PLAN,
        )
        materializer.register(system)
        materializer.register(plan)

        group = materializer.build_group(
            group_id="prompt-1",
            workflow_id="wf",
            consumer="planner",
            ordered_segment_ids=["system-1", "plan-1"],
            prompt_id="planner-1",
        )
        decision = materializer.prepare_group_read(
            group,
            policy=WeakHeuristicSegmentPolicy(),
        )

        self.assertIn("system-1", decision.pin_in_hbm)
        request = materializer.build_vllm_request(group)
        self.assertEqual(
            request.assembled_prompt(),
            "[SYSTEM]\nYou are a planner.\n\n[PLAN]\nPlan v1",
        )
        snapshot = materializer.snapshot()
        self.assertEqual(snapshot.segment_states["system-1"], ContextSegmentState.RESIDENT)
        self.assertEqual(snapshot.segment_states["plan-1"], ContextSegmentState.RESIDENT)
        self.assertGreaterEqual(snapshot.hbm_bytes, len("Plan v1".encode("utf-8")))

    def test_segmented_generation_request_places_dynamic_user_prompt_last(self):
        system = self.make_segment(
            state_id="system-1",
            logical_key="prompt/system",
            module="system",
            version=1,
            text="You are a planner.",
            role=SegmentRole.SYSTEM,
            is_shared=True,
            is_immutable=True,
        )
        task = self.make_segment(
            state_id="task-1",
            logical_key="prompt/task",
            module="task",
            version=1,
            text="Fix bug #123.",
            role=SegmentRole.TASK,
        )
        plan = self.make_segment(
            state_id="plan-1",
            logical_key="planner/plan",
            module="plan",
            version=1,
            text="1. Inspect failing path.\n2. Patch logic.",
            role=SegmentRole.PLAN,
        )
        request = RUNTIME.SegmentedGenerationRequest(
            ordered_segments=(system, task, plan),
            request_segments=(task, plan),
            fallback_system_prompt=system.text,
            fallback_user_prompt="Iteration: 2\nCurrent phase: coding",
        )

        messages = request.to_openai_messages()

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a planner.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertTrue(messages[1]["content"].startswith("[TASK]\nFix bug #123."))
        self.assertIn("[PLAN]\n1. Inspect failing path.", messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith("Current phase: coding"))

    def test_segmented_generation_request_reports_payload_rows(self):
        task = self.make_segment(
            state_id="task-1",
            logical_key="prompt/task",
            module="task",
            version=1,
            text="Fix bug #123.",
            role=SegmentRole.TASK,
        )
        evidence = self.make_segment(
            state_id="evidence-1",
            logical_key="retrieval/readme",
            module="evidence",
            version=1,
            text="Fix bug #123.\nREADME note.",
            role=SegmentRole.EVIDENCE,
        )
        request = RUNTIME.SegmentedGenerationRequest(
            ordered_segments=(task, evidence),
            request_segments=(task, evidence),
            fallback_system_prompt="You are a coder.",
            fallback_user_prompt="Iteration: 1\nCurrent phase: coding",
        )

        rows = request.payload_rows()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["role"], "task")
        self.assertEqual(rows[1]["role"], "evidence")
        self.assertGreater(rows[1]["serialized_token_count"], 0)
        self.assertGreaterEqual(rows[1]["overlap_token_estimate"], 0)
        self.assertGreaterEqual(
            rows[1]["cumulative_serialized_tokens"],
            rows[0]["serialized_token_count"],
        )

    def test_policy_offloads_ephemeral_inactive_hbm_segments(self):
        materializer = SegmentMaterializer(serving_adapter=SyntheticMaterializationAdapter())
        scratch = self.make_segment(
            state_id="scratch-1",
            logical_key="scratch/notes",
            module="scratchpad",
            version=1,
            text="old scratch",
            role=SegmentRole.SCRATCH,
            is_ephemeral=True,
        )
        materializer.register_segment(scratch, initial_tier=ResidencyTier.HBM)
        materializer.record_read(["scratch-1"])
        group = materializer.build_group(
            group_id="prompt-2",
            workflow_id="wf",
            consumer="planner",
            ordered_segment_ids=[],
        )

        decision = WeakHeuristicSegmentPolicy().decide(
            materializer=materializer,
            group=group,
        )

        self.assertIn("scratch-1", decision.offload_to_cpu)

    def test_release_is_semantic_and_reclaim_is_delegated(self):
        adapter = SyntheticMaterializationAdapter()
        materializer = SegmentMaterializer(serving_adapter=adapter)
        evidence = self.make_segment(
            state_id="evidence-1",
            logical_key="retrieval/doc-1",
            module="evidence",
            version=1,
            text="doc",
            role=SegmentRole.EVIDENCE,
        )
        materializer.register(evidence)
        context = self.make_context(segment_id="evidence-1")
        lookup = materializer.lookup(evidence, context)

        with self.assertRaises(ReclaimNotSupportedError):
            adapter.reclaim(lookup.association)

        materializer.release(evidence)
        self.assertEqual(materializer.get_record("evidence-1").semantic_state, SemanticState.RELEASED)
        self.assertFalse(materializer.get_record("evidence-1").associations[context].resident)


if __name__ == "__main__":
    unittest.main()
