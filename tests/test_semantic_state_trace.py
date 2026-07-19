import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "swebench" / "inference" / "trace" / "semantic_state.py"
MODULE_SPEC = importlib.util.spec_from_file_location("semantic_state", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Failed to load semantic_state module from {MODULE_PATH}")
SEMANTIC_STATE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = SEMANTIC_STATE
MODULE_SPEC.loader.exec_module(SEMANTIC_STATE)

TraceLogger = SEMANTIC_STATE.TraceLogger
TraceLoggerError = SEMANTIC_STATE.TraceLoggerError
get_schema_path = SEMANTIC_STATE.get_schema_path


def load_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TraceLoggerTests(unittest.TestCase):
    def test_trace_logger_emits_versioned_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"

            with TraceLogger(
                trace_path,
                workflow_id="task-1",
                tenant_id="tenant-a",
            ) as logger:
                plan_v1 = logger.create_state(
                    state_id="plan_v1",
                    logical_key="plan",
                    state_type="plan",
                    size_bytes=128,
                    token_count=32,
                    producer="planner",
                    materialization="HBM",
                )
                logger.log_prompt_assembly(
                    consumer="coder",
                    state_ids=[plan_v1.state_id],
                    prompt_id="prompt-1",
                )
                plan_v2 = logger.create_state(
                    state_id="plan_v2",
                    logical_key="plan",
                    state_type="plan",
                    size_bytes=144,
                    token_count=36,
                    producer="planner",
                    parent_state_ids=[plan_v1.state_id],
                    materialization="HBM",
                )
                logger.supersede_state(plan_v1.state_id, plan_v2.state_id)
                logger.release_state(plan_v1.state_id, consumer="coder")

            events = load_events(trace_path)

        self.assertEqual(
            [event["op"] for event in events],
            ["CREATE", "READ", "DERIVE", "SUPERSEDE", "RELEASE"],
        )
        self.assertEqual([event["ts"] for event in events], [0, 1, 2, 3, 4])
        self.assertEqual(events[0]["version"], 1)
        self.assertEqual(events[1]["metadata"]["prompt_id"], "prompt-1")
        self.assertEqual(events[2]["version"], 2)
        self.assertEqual(events[2]["supersedes"], "plan_v1")
        self.assertEqual(events[2]["parent_state_ids"], ["plan_v1"])
        self.assertEqual(events[3]["old_state_id"], "plan_v1")
        self.assertEqual(events[3]["new_state_id"], "plan_v2")

    def test_trace_logger_tracks_prompt_reads_and_tiers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            timestamps = iter([10, 10, 11, 15, 16, 20])

            with TraceLogger(
                trace_path,
                workflow_id="task-2",
                tenant_id="tenant-b",
                clock=lambda: next(timestamps),
            ) as logger:
                doc = logger.create_state(
                    state_id="doc_1",
                    logical_key="retrieval/doc.py",
                    state_type="retrieved_document",
                    size_bytes=256,
                    token_count=64,
                    producer="retriever",
                    materialization="CPU",
                )
                logger.materialize_state(doc.state_id, tier="HBM")
                logger.log_prompt_assembly(
                    consumer="coder",
                    state_ids=[doc.state_id],
                    prompt_id="prompt-2",
                    metadata={"hook": "messages_for_llm"},
                )
                logger.evict_state(doc.state_id)
                logger.reload_state(doc.state_id, from_tier="CPU", to_tier="HBM")
                logger.release_state(doc.state_id, consumer="coder")

            events = load_events(trace_path)

        self.assertEqual([event["ts"] for event in events], [10, 10, 11, 15, 16, 20])
        self.assertEqual(events[1]["op"], "MATERIALIZE")
        self.assertEqual(events[1]["tier"], "HBM")
        self.assertEqual(events[2]["metadata"]["hook"], "messages_for_llm")
        self.assertEqual(events[4]["from_tier"], "CPU")
        self.assertEqual(events[4]["to_tier"], "HBM")

    def test_trace_logger_rejects_unknown_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"

            with TraceLogger(
                trace_path,
                workflow_id="task-3",
                tenant_id="tenant-c",
            ) as logger:
                with self.assertRaisesRegex(TraceLoggerError, "unknown state_id"):
                    logger.create_state(
                        state_id="patch_v1",
                        logical_key="patch",
                        state_type="generated_artifact",
                        size_bytes=100,
                        token_count=25,
                        producer="coder",
                        parent_state_ids=["missing_state"],
                    )

    def test_schema_is_packaged(self):
        self.assertTrue(get_schema_path().is_file())
