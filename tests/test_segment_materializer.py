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
ResidencyTier = RUNTIME.ResidencyTier
SegmentIdentity = RUNTIME.SegmentIdentity
SegmentMaterializer = RUNTIME.SegmentMaterializer
SegmentVersion = RUNTIME.SegmentVersion
WeakHeuristicSegmentPolicy = RUNTIME.WeakHeuristicSegmentPolicy


class SegmentMaterializerTests(unittest.TestCase):
    def test_prepare_group_read_promotes_and_assembles_segments(self):
        materializer = SegmentMaterializer()
        materializer.register_segment(
            SegmentVersion(
                state_id="system-1",
                identity=SegmentIdentity(logical_key="system", module="system"),
                version=1,
                size_bytes=100,
                token_count=20,
                text="You are a planner.",
                is_shared=True,
                residency_hint=ResidencyTier.HBM,
            )
        )
        materializer.register_segment(
            SegmentVersion(
                state_id="plan-1",
                identity=SegmentIdentity(logical_key="plan", module="plan"),
                version=1,
                size_bytes=40,
                token_count=10,
                text="Plan v1",
                is_ephemeral=True,
                residency_hint=ResidencyTier.EVICTED,
            )
        )

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

        self.assertIn("plan-1", decision.rematerialize_for_read)
        request = materializer.build_vllm_request(group)
        self.assertEqual(
            request.assembled_prompt(),
            "You are a planner.\n\nPlan v1",
        )

        snapshot = materializer.snapshot()
        self.assertEqual(snapshot.segment_tiers["system-1"], ResidencyTier.HBM)
        self.assertEqual(snapshot.segment_tiers["plan-1"], ResidencyTier.CPU)
        self.assertEqual(snapshot.access_counts["plan-1"], 1)

    def test_policy_offloads_ephemeral_inactive_hbm_segments(self):
        materializer = SegmentMaterializer()
        materializer.register_segment(
            SegmentVersion(
                state_id="scratch-1",
                identity=SegmentIdentity(logical_key="scratch", module="scratchpad"),
                version=1,
                size_bytes=25,
                token_count=5,
                text="old scratch",
                is_ephemeral=True,
                residency_hint=ResidencyTier.HBM,
            )
        )
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
