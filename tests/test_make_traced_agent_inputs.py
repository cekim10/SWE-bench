import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "swebench" / "inference" / "make_traced_agent_inputs.py"


def load_converter_module():
    for package_name in [
        "swebench",
        "swebench.inference",
        "swebench.inference.make_datasets",
    ]:
        if package_name not in sys.modules:
            module = types.ModuleType(package_name)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[package_name] = module

    create_instance = types.ModuleType("swebench.inference.make_datasets.create_instance")
    create_instance.PROMPT_FUNCTIONS = {"style-3": object(), "style-2": object()}

    def add_text_inputs(*args, **kwargs):
        raise AssertionError("add_text_inputs should not be called in this unit test")

    create_instance.add_text_inputs = add_text_inputs
    sys.modules["swebench.inference.make_datasets.create_instance"] = create_instance

    tokenize_dataset = types.ModuleType(
        "swebench.inference.make_datasets.tokenize_dataset"
    )
    tokenize_dataset.TOKENIZER_FUNCS = {"cl100k": object()}
    sys.modules[
        "swebench.inference.make_datasets.tokenize_dataset"
    ] = tokenize_dataset

    spec = importlib.util.spec_from_file_location(
        "swebench.inference.make_traced_agent_inputs",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load make_traced_agent_inputs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONVERTER = load_converter_module()


class MakeTracedAgentInputsTests(unittest.TestCase):
    def test_processed_to_traced_instance_keeps_expected_fields(self):
        traced = CONVERTER.processed_to_traced_instance(
            {
                "instance_id": "demo-1",
                "problem_statement": "Fix the bug.",
                "file_contents": {"foo.py": "print('x')\n"},
                "readmes": {"README.md": "docs"},
                "repo": "org/repo",
                "base_commit": "abc123",
                "text_inputs": "prompt",
            }
        )
        self.assertEqual(traced["instance_id"], "demo-1")
        self.assertIn("foo.py", traced["file_contents"])
        self.assertEqual(traced["metadata"]["repo"], "org/repo")
        self.assertTrue(traced["metadata"]["source_text_inputs_present"])

    def test_convert_processed_file_applies_limit_and_sort(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_path = Path(tmpdir) / "processed.jsonl"
            output_path = Path(tmpdir) / "traced.jsonl"
            rows = [
                {
                    "instance_id": "b-task",
                    "problem_statement": "B",
                    "file_contents": {"b.py": "b\n"},
                },
                {
                    "instance_id": "a-task",
                    "problem_statement": "A",
                    "file_contents": {"a.py": "a\n"},
                },
            ]
            processed_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            count = CONVERTER.convert_processed_file(processed_path, output_path, limit=1)
            self.assertEqual(count, 1)
            output_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(output_rows[0]["instance_id"], "a-task")

    def test_select_instance_records_filters_ids(self):
        selected = CONVERTER.select_instance_records(
            [
                {"instance_id": "x", "problem_statement": "X"},
                {"instance_id": "y", "problem_statement": "Y"},
            ],
            instance_ids=["y"],
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["instance_id"], "y")

