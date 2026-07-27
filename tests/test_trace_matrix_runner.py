import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "swebench" / "inference" / "trace"
INFERENCE_DIR = ROOT / "swebench" / "inference"
RUNTIME_DIR = ROOT / "swebench" / "inference" / "runtime"


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
        ("swebench.inference.trace.semantic_state", TRACE_DIR / "semantic_state.py"),
        ("swebench.inference.trace.agentic", TRACE_DIR / "agentic.py"),
        ("swebench.inference.trace.analysis", TRACE_DIR / "analysis.py"),
        ("swebench.inference.run_traced_agent_matrix", INFERENCE_DIR / "run_traced_agent_matrix.py"),
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
MATRIX = MODULES["swebench.inference.run_traced_agent_matrix"]


class NamedStubBackend(AGENTIC.StubModelBackend):
    def __init__(self, name: str) -> None:
        self.name = name


class TraceMatrixRunnerTests(unittest.TestCase):
    def test_make_backend_supports_ollama_groq_and_continuum(self):
        ollama_backend = AGENTIC.make_backend("ollama", "qwen2.5-coder")
        groq_backend = AGENTIC.make_backend("groq", "llama-3.3-70b-versatile")
        vllm_backend = AGENTIC.make_backend("vllm", "Qwen/Qwen2.5-Coder-3B-Instruct")
        continuum_backend = AGENTIC.make_backend(
            "continuum", "Qwen/Qwen2.5-Coder-3B-Instruct"
        )
        self.assertEqual(ollama_backend.name, "ollama")
        self.assertEqual(groq_backend.name, "groq")
        self.assertEqual(vllm_backend.name, "vllm")
        self.assertEqual(continuum_backend.name, "continuum")
        self.assertEqual(
            continuum_backend.runtime_inference_config(
                step_name="planner",
                prompt_mode="monolithic",
            )["transport"],
            "openai-compatible-continuum",
        )

    def test_matrix_runner_emits_provider_breakdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "matrix"

            def fake_make_backend(provider, model, **kwargs):
                return NamedStubBackend(provider)

            argv = [
                "run_traced_agent_matrix.py",
                "--demo_instance",
                "--provider_spec",
                "stub",
                "--provider_spec",
                "ollama:qwen2.5-coder",
                "--provider_spec",
                "groq:llama-3.3-70b-versatile",
                "--output_dir",
                str(output_dir),
            ]
            with mock.patch.object(MATRIX, "make_backend", side_effect=fake_make_backend):
                with mock.patch.object(sys, "argv", argv):
                    MATRIX.main()

            combined_path = output_dir / "combined.jsonl"
            analysis_path = output_dir / "analysis.json"
            self.assertTrue(combined_path.exists())
            self.assertTrue(analysis_path.exists())

            records = [
                json.loads(line)
                for line in combined_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 3)
            self.assertEqual(
                sorted(record["provider"] for record in records),
                ["groq", "ollama", "stub"],
            )

            report = json.loads(analysis_path.read_text(encoding="utf-8"))
            self.assertEqual(report["run_summary"]["provider_counts"]["stub"], 1)
            self.assertEqual(report["run_summary"]["provider_counts"]["ollama"], 1)
            self.assertEqual(report["run_summary"]["provider_counts"]["groq"], 1)
            self.assertEqual(report["trace_count"], 0)
