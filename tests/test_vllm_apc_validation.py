import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = ROOT / "swebench" / "inference"
RUNTIME_DIR = INFERENCE_DIR / "runtime"


def load_modules():
    for package_name in [
        "swebench",
        "swebench.inference",
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
        ("swebench.inference.validate_vllm_apc", INFERENCE_DIR / "validate_vllm_apc.py"),
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
APC = MODULES["swebench.inference.validate_vllm_apc"]
VLLM = MODULES["swebench.inference.runtime.vllm_adapter"]


class FakeAPCAdapter:
    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def complete_with_details(self, request):
        digest = request.message_digest()
        seen_count = self.seen.get(digest, 0)
        self.seen[digest] = seen_count + 1
        if seen_count:
            duration_ms = 20.0
        elif request.extra_body.get("probe_case") == "shared_prefix_variant":
            duration_ms = 40.0
        else:
            duration_ms = 80.0
        return VLLM.VLLMCompletionResult(
            text=f"ok:{request.extra_body.get('probe_case')}",
            duration_ms=duration_ms,
            message_digest=digest,
            prompt_tokens=32,
            completion_tokens=8,
            total_tokens=40,
        )


class VLLMAPCValidationTests(unittest.TestCase):
    def test_build_probe_cases_is_deterministic(self):
        cases = APC.build_probe_cases(model="Qwen/Test-Model", max_tokens=8)
        self.assertEqual([case.name for case in cases], [
            "cold_same_prompt",
            "warm_same_prompt",
            "shared_prefix_variant",
        ])
        self.assertEqual(
            cases[0].request.message_digest(),
            cases[1].request.message_digest(),
        )
        self.assertNotEqual(
            cases[0].request.message_digest(),
            cases[2].request.message_digest(),
        )
        self.assertEqual(cases[0].prefix_digest, cases[1].prefix_digest)
        self.assertEqual(cases[0].prefix_digest, cases[2].prefix_digest)

    def test_run_probe_cases_reports_same_prompt_and_shared_prefix_behavior(self):
        cases = APC.build_probe_cases(model="Qwen/Test-Model", max_tokens=8)
        report = APC.run_probe_cases(adapter=FakeAPCAdapter(), cases=cases)

        self.assertEqual(len(report["observations"]), 3)
        self.assertTrue(report["summary"]["same_prompt_message_match"])
        self.assertTrue(report["summary"]["shared_prefix_digest_match"])
        self.assertLess(report["summary"]["same_prompt_duration_ratio"], 1.0)
        self.assertLess(report["summary"]["shared_prefix_variant_duration_ratio"], 1.0)
        self.assertTrue(report["summary"]["observed_same_prompt_speedup"])
        self.assertTrue(report["summary"]["observed_shared_prefix_speedup"])


if __name__ == "__main__":
    unittest.main()
