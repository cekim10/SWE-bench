import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from textwrap import dedent


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
            super().__init__()
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

    def test_register_runtime_segment_preserves_explicit_role(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AGENTIC.TracedAgentRunner(
                backend=AGENTIC.StubModelBackend(),
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                prompt_runtime_mode="segment_aware",
            )
            materializer = AGENTIC.SegmentRuntime()
            handle = types.SimpleNamespace(
                state_id="task_v1",
                logical_key="prompt/task",
                version=1,
            )

            runner._register_runtime_segment(
                materializer=materializer,
                handle=handle,
                workflow_id="wf",
                module="task",
                role="task",
                text="Fix bug #123",
                materialization="HBM",
                is_shared=True,
                is_immutable=True,
                is_ephemeral=False,
            )

            self.assertEqual(
                runner._runtime_segment_cache["task_v1"].role.value,
                "task",
            )

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

    def test_synthetic_stress_runner_emits_runtime_and_backend_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = self.CapturingBackend()
            runner = AGENTIC.SyntheticStressTracedAgentRunner(
                backend=backend,
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=3,
                max_files=2,
                prompt_runtime_mode="segment_aware",
            )
            result = runner.run_instance(AGENTIC.build_demo_instance())
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["agent_family"], "synthetic_stress")
            self.assertTrue(result["trace_validation"]["is_valid"])
            self.assertTrue(Path(result["runtime_event_path"]).exists())
            self.assertTrue(Path(result["backend_call_path"]).exists())
            self.assertGreater(result["backend_call_summary"]["request_count"], 0)
            self.assertGreater(len(backend.calls), 0)

    def test_deep_research_runner_emits_runtime_and_backend_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = self.CapturingBackend()
            runner = AGENTIC.DeepResearchTracedAgentRunner(
                backend=backend,
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=3,
                max_files=3,
                prompt_runtime_mode="segment_aware",
            )
            result = runner.run_instance(AGENTIC.build_deep_research_demo_instance())
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["agent_family"], "deep_research")
            self.assertTrue(result["trace_validation"]["is_valid"])
            self.assertTrue(Path(result["runtime_event_path"]).exists())
            self.assertTrue(Path(result["backend_call_path"]).exists())
            self.assertGreater(result["backend_call_summary"]["request_count"], 0)
            self.assertGreater(len(backend.calls), 0)

            events = AGENTIC.load_trace_events(result["trace_path"])
            producers = {event.get("producer") for event in events if "producer" in event}
            logical_keys = {event.get("logical_key") for event in events if "logical_key" in event}
            self.assertIn("research_planner", producers)
            self.assertIn("research_reader", producers)
            self.assertIn("research_writer", producers)
            self.assertIn("research_critic", producers)
            self.assertIn("research/planner/plan", logical_keys)
            self.assertIn("research/writer/draft", logical_keys)
            self.assertIn("research/critic/feedback", logical_keys)

    def test_load_open_deep_research_prompt_pack_from_repo_checkout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "open_deep_research"
            prompts_dir = repo_root / "src" / "open_deep_research"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            (repo_root / "langgraph.json").write_text(
                '{"graphs": {"Deep Researcher": "./src/open_deep_research/deep_researcher.py:deep_researcher"}}',
                encoding="utf-8",
            )
            (prompts_dir / "prompts.py").write_text(
                dedent(
                    '''
                    clarify_with_user_instructions = "clarify {messages} {date}"
                    transform_messages_into_research_topic_prompt = "brief {messages} {date}"
                    lead_researcher_prompt = "lead {date} {max_concurrent_research_units} {max_researcher_iterations}"
                    research_system_prompt = "research {mcp_prompt} {date}"
                    compress_research_system_prompt = "compress {date}"
                    compress_research_simple_human_message = "cleanup"
                    final_report_generation_prompt = "final {research_brief} {messages} {findings}"
                    '''
                ),
                encoding="utf-8",
            )
            prompt_pack = AGENTIC.load_open_deep_research_prompt_pack(repo_root)
            self.assertEqual(prompt_pack.repo_path, repo_root.resolve())
            self.assertIn("lead", prompt_pack.lead_researcher_prompt)

    def test_open_deep_research_runner_uses_upstream_prompt_pack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "open_deep_research"
            prompts_dir = repo_root / "src" / "open_deep_research"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            (repo_root / "langgraph.json").write_text(
                '{"graphs": {"Deep Researcher": "./src/open_deep_research/deep_researcher.py:deep_researcher"}}',
                encoding="utf-8",
            )
            (prompts_dir / "prompts.py").write_text(
                dedent(
                    '''
                    clarify_with_user_instructions = "clarify {messages} {date}"
                    transform_messages_into_research_topic_prompt = "brief {messages} {date}"
                    lead_researcher_prompt = "UPSTREAM_SUPERVISOR {date} {max_concurrent_research_units} {max_researcher_iterations}"
                    research_system_prompt = "UPSTREAM_RESEARCH {mcp_prompt} {date}"
                    compress_research_system_prompt = "UPSTREAM_COMPRESS {date}"
                    compress_research_simple_human_message = "UPSTREAM_CLEANUP"
                    final_report_generation_prompt = "UPSTREAM_FINAL {research_brief} {messages} {findings}"
                    '''
                ),
                encoding="utf-8",
            )

            backend = self.CapturingBackend()
            runner = AGENTIC.OpenDeepResearchTracedAgentRunner(
                backend=backend,
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
                prompt_runtime_mode="segment_aware",
                open_deep_research_path=repo_root,
            )
            result = runner.run_instance(AGENTIC.build_deep_research_demo_instance())
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["workload_source"], "open_deep_research")
            self.assertEqual(
                Path(result["open_deep_research_path"]),
                repo_root.resolve(),
            )
            self.assertGreater(len(backend.calls), 0)
            self.assertTrue(
                any(
                    "UPSTREAM_SUPERVISOR" in call.get("system_prompt", "")
                    for call in backend.calls
                )
            )
            self.assertTrue(
                any(
                    "UPSTREAM_RESEARCH" in call.get("system_prompt", "")
                    for call in backend.calls
                )
            )

    def test_open_deep_research_monolithic_flattens_active_context_for_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "open_deep_research"
            prompts_dir = repo_root / "src" / "open_deep_research"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            (repo_root / "langgraph.json").write_text(
                '{"graphs": {"Deep Researcher": "./src/open_deep_research/deep_researcher.py:deep_researcher"}}',
                encoding="utf-8",
            )
            (prompts_dir / "prompts.py").write_text(
                dedent(
                    '''
                    clarify_with_user_instructions = "clarify {messages} {date}"
                    transform_messages_into_research_topic_prompt = "brief {messages} {date}"
                    lead_researcher_prompt = "UPSTREAM_SUPERVISOR {date} {max_concurrent_research_units} {max_researcher_iterations}"
                    research_system_prompt = "UPSTREAM_RESEARCH {mcp_prompt} {date}"
                    compress_research_system_prompt = "UPSTREAM_COMPRESS {date}"
                    compress_research_simple_human_message = "UPSTREAM_CLEANUP"
                    final_report_generation_prompt = "UPSTREAM_FINAL {research_brief} {messages} {findings} {date}"
                    '''
                ),
                encoding="utf-8",
            )

            backend = self.CapturingBackend()
            runner = AGENTIC.OpenDeepResearchTracedAgentRunner(
                backend=backend,
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
                prompt_runtime_mode="monolithic",
                open_deep_research_path=repo_root,
            )
            result = runner.run_instance(AGENTIC.build_deep_research_demo_instance())
            self.assertEqual(result["status"], "COMPLETED")
            self.assertGreater(len(backend.calls), 0)
            self.assertTrue(
                all(call.get("prompt_mode") == "monolithic" for call in backend.calls)
            )
            self.assertTrue(
                all(call.get("segment_request") is not None for call in backend.calls)
            )

    def test_load_swe_agent_prompt_pack_from_repo_checkout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "swe_agent"
            config_dir = repo_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "default.yaml").write_text(
                dedent(
                    """
                    agent:
                      templates:
                        system_template: |-
                          UPSTREAM_SWE_SYSTEM
                        instance_template: |-
                          repo {{working_dir}}
                          issue {{problem_statement}}
                        next_step_template: |-
                          OBSERVATION:
                          {{observation}}
                        next_step_no_output_template: |-
                          no output
                      tools:
                        registry_variables:
                          SUBMIT_REVIEW_MESSAGES:
                            - |
                              review {{diff}}
                    """
                ),
                encoding="utf-8",
            )
            prompt_pack = AGENTIC.load_swe_agent_prompt_pack(repo_root)
            self.assertEqual(prompt_pack.repo_path, repo_root.resolve())
            self.assertIn("UPSTREAM_SWE_SYSTEM", prompt_pack.system_template)
            self.assertEqual(len(prompt_pack.submit_review_messages), 1)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "langgraph not installed in this interpreter")
    def test_open_swe_agent_runner_uses_upstream_prompt_pack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "swe_agent"
            config_dir = repo_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "default.yaml").write_text(
                dedent(
                    """
                    agent:
                      templates:
                        system_template: |-
                          UPSTREAM_SWE_SYSTEM
                        instance_template: |-
                          repo {{working_dir}}
                          issue {{problem_statement}}
                        next_step_template: |-
                          OBSERVATION:
                          {{observation}}
                        next_step_no_output_template: |-
                          no output
                      tools:
                        registry_variables:
                          SUBMIT_REVIEW_MESSAGES:
                            - |
                              review {{diff}}
                    """
                ),
                encoding="utf-8",
            )

            backend = self.CapturingBackend()
            runner = AGENTIC.OpenSWEAgentTracedAgentRunner(
                backend=backend,
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
                prompt_runtime_mode="segment_aware",
                swe_agent_path=repo_root,
            )
            result = runner.run_instance(AGENTIC.build_demo_instance())
            self.assertEqual(result["agent_family"], "swe_agent")
            self.assertEqual(result["workload_source"], "swe_agent")
            self.assertEqual(Path(result["swe_agent_path"]), repo_root.resolve())
            self.assertGreater(len(backend.calls), 0)
            self.assertTrue(
                any(
                    "UPSTREAM_SWE_SYSTEM" in call.get("system_prompt", "")
                    for call in backend.calls
                )
            )

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "langgraph not installed in this interpreter")
    def test_open_swe_agent_monolithic_flattens_active_context_for_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "swe_agent"
            config_dir = repo_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "default.yaml").write_text(
                dedent(
                    """
                    agent:
                      templates:
                        system_template: |-
                          UPSTREAM_SWE_SYSTEM
                        instance_template: |-
                          repo {{working_dir}}
                          issue {{problem_statement}}
                        next_step_template: |-
                          OBSERVATION:
                          {{observation}}
                        next_step_no_output_template: |-
                          no output
                      tools:
                        registry_variables:
                          SUBMIT_REVIEW_MESSAGES:
                            - |
                              review {{diff}}
                    """
                ),
                encoding="utf-8",
            )

            backend = self.CapturingBackend()
            runner = AGENTIC.OpenSWEAgentTracedAgentRunner(
                backend=backend,
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
                prompt_runtime_mode="monolithic",
                swe_agent_path=repo_root,
            )
            result = runner.run_instance(AGENTIC.build_demo_instance())
            self.assertEqual(result["agent_family"], "swe_agent")
            self.assertGreater(len(backend.calls), 0)
            self.assertTrue(
                all(call.get("prompt_mode") == "monolithic" for call in backend.calls)
            )
            self.assertTrue(
                all(call.get("segment_request") is not None for call in backend.calls)
            )

    def test_load_openhands_prompt_pack_from_repo_checkout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "openhands"
            codeact_dir = repo_root / "openhands" / "agenthub" / "codeact_agent"
            codeact_dir.mkdir(parents=True, exist_ok=True)
            (repo_root / "config.template.toml").write_text(
                dedent(
                    """
                    #default_agent = "CodeActAgent"
                    [agent]
                    enable_editor = true
                    """
                ),
                encoding="utf-8",
            )
            (repo_root / "AGENTS.md").write_text(
                "UPSTREAM_OPENHANDS_GUIDE\n\nRun tests with pytest.\n",
                encoding="utf-8",
            )
            (repo_root / "Development.md").write_text(
                "UPSTREAM_OPENHANDS_DEV\n\nUse make build for setup.\n",
                encoding="utf-8",
            )
            (codeact_dir / "codeact_agent.py").write_text(
                "class CodeActAgent:\n    pass\n",
                encoding="utf-8",
            )

            prompt_pack = AGENTIC.load_openhands_prompt_pack(repo_root)
            self.assertEqual(prompt_pack.repo_path, repo_root.resolve())
            self.assertEqual(prompt_pack.default_agent_name, "CodeActAgent")
            self.assertIn("UPSTREAM_OPENHANDS_GUIDE", prompt_pack.agents_guidance)

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "langgraph not installed in this interpreter")
    def test_openhands_runner_uses_upstream_prompt_pack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "openhands"
            codeact_dir = repo_root / "openhands" / "agenthub" / "codeact_agent"
            codeact_dir.mkdir(parents=True, exist_ok=True)
            (repo_root / "config.template.toml").write_text(
                dedent(
                    """
                    #default_agent = "CodeActAgent"
                    [agent]
                    enable_editor = true
                    """
                ),
                encoding="utf-8",
            )
            (repo_root / "AGENTS.md").write_text(
                "UPSTREAM_OPENHANDS_GUIDE\n\nPrefer pytest and repository-aware edits.\n",
                encoding="utf-8",
            )
            (repo_root / "Development.md").write_text(
                "UPSTREAM_OPENHANDS_DEV\n\nUse make build for setup.\n",
                encoding="utf-8",
            )
            (codeact_dir / "codeact_agent.py").write_text(
                "class CodeActAgent:\n    pass\n",
                encoding="utf-8",
            )

            backend = self.CapturingBackend()
            runner = AGENTIC.OpenHandsTracedAgentRunner(
                backend=backend,
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
                prompt_runtime_mode="segment_aware",
                openhands_path=repo_root,
            )
            result = runner.run_instance(AGENTIC.build_demo_instance())
            self.assertEqual(result["agent_family"], "openhands")
            self.assertEqual(result["workload_source"], "openhands")
            self.assertEqual(Path(result["openhands_path"]), repo_root.resolve())
            self.assertGreater(len(backend.calls), 0)
            self.assertTrue(
                any(
                    "UPSTREAM_OPENHANDS_GUIDE" in call.get("system_prompt", "")
                    for call in backend.calls
                )
            )
            self.assertTrue(
                any(
                    "CodeActAgent" in call.get("system_prompt", "")
                    for call in backend.calls
                )
            )

    @unittest.skipUnless(importlib.util.find_spec("langgraph"), "langgraph not installed in this interpreter")
    def test_openhands_monolithic_flattens_active_context_for_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "openhands"
            codeact_dir = repo_root / "openhands" / "agenthub" / "codeact_agent"
            codeact_dir.mkdir(parents=True, exist_ok=True)
            (repo_root / "config.template.toml").write_text(
                dedent(
                    """
                    #default_agent = "CodeActAgent"
                    [agent]
                    enable_editor = true
                    """
                ),
                encoding="utf-8",
            )
            (repo_root / "AGENTS.md").write_text(
                "UPSTREAM_OPENHANDS_GUIDE\n\nPrefer pytest and repository-aware edits.\n",
                encoding="utf-8",
            )
            (repo_root / "Development.md").write_text(
                "UPSTREAM_OPENHANDS_DEV\n\nUse make build for setup.\n",
                encoding="utf-8",
            )
            (codeact_dir / "codeact_agent.py").write_text(
                "class CodeActAgent:\n    pass\n",
                encoding="utf-8",
            )

            backend = self.CapturingBackend()
            runner = AGENTIC.OpenHandsTracedAgentRunner(
                backend=backend,
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
                prompt_runtime_mode="monolithic",
                openhands_path=repo_root,
            )
            result = runner.run_instance(AGENTIC.build_demo_instance())
            self.assertEqual(result["agent_family"], "openhands")
            self.assertGreater(len(backend.calls), 0)
            self.assertTrue(
                all(call.get("prompt_mode") == "monolithic" for call in backend.calls)
            )
            self.assertTrue(
                all(call.get("segment_request") is not None for call in backend.calls)
            )

    def test_openhands_request_segment_selection_prefers_semantic_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "openhands"
            codeact_dir = repo_root / "openhands" / "agenthub" / "codeact_agent"
            codeact_dir.mkdir(parents=True, exist_ok=True)
            (repo_root / "config.template.toml").write_text(
                '#default_agent = "CodeActAgent"\n',
                encoding="utf-8",
            )
            (repo_root / "AGENTS.md").write_text("guide\n", encoding="utf-8")
            (codeact_dir / "codeact_agent.py").write_text(
                "class CodeActAgent:\n    pass\n",
                encoding="utf-8",
            )

            runner = AGENTIC.OpenHandsTracedAgentRunner(
                backend=AGENTIC.StubModelBackend(),
                trace_dir=Path(tmpdir) / "traces",
                tenant_id="test-tenant",
                max_iterations=2,
                max_files=2,
                prompt_runtime_mode="segment_aware",
                openhands_path=repo_root,
            )
            reviewer_segments = [
                {"state_id": "system_v1", "segment_role": "system"},
                {"state_id": "task_v1", "segment_role": "task"},
                {"state_id": "plan_v1", "segment_role": "plan"},
                {"state_id": "patch_v1", "segment_role": "artifact"},
                {"state_id": "review_v1", "segment_role": "review"},
            ]
            planner_segments = [
                {"state_id": "system_v1", "segment_role": "system", "token_count": 20},
                {"state_id": "task_v1", "segment_role": "task", "token_count": 40},
                {"state_id": "route_v1", "segment_role": "router", "token_count": 24},
                {"state_id": "readme_1", "segment_role": "evidence", "token_count": 1200},
                {"state_id": "retrieval_1", "segment_role": "evidence", "token_count": 250},
                {"state_id": "retrieval_2", "segment_role": "evidence", "token_count": 300},
            ]
            coder_segments = [
                {"state_id": "system_v1", "segment_role": "system", "token_count": 20},
                {"state_id": "task_v1", "segment_role": "task", "token_count": 40},
                {"state_id": "route_v1", "segment_role": "router", "token_count": 24},
                {"state_id": "plan_v1", "segment_role": "plan", "token_count": 36},
                {"state_id": "readme_1", "segment_role": "evidence", "token_count": 1200},
                {"state_id": "retrieval_1", "segment_role": "evidence", "token_count": 250},
                {"state_id": "patch_v1", "segment_role": "artifact", "token_count": 80},
            ]
            tester_segments = [
                {"state_id": "system_v1", "segment_role": "system", "token_count": 20},
                {"state_id": "task_v1", "segment_role": "task", "token_count": 40},
                {"state_id": "patch_v1", "segment_role": "artifact", "token_count": 80},
                {"state_id": "review_v1", "segment_role": "review", "token_count": 32},
            ]

            planner_request_ids = runner._request_segment_ids_for_step(
                step_name="planner",
                segments=planner_segments,
            )
            coder_request_ids = runner._request_segment_ids_for_step(
                step_name="coder",
                segments=coder_segments,
            )
            reviewer_request_ids = runner._request_segment_ids_for_step(
                step_name="reviewer",
                segments=reviewer_segments,
            )
            tester_request_ids = runner._request_segment_ids_for_step(
                step_name="tester",
                segments=tester_segments,
            )

            self.assertEqual(
                planner_request_ids,
                ["task_v1", "route_v1", "retrieval_1", "retrieval_2"],
            )
            self.assertEqual(
                coder_request_ids,
                ["task_v1", "route_v1", "plan_v1", "retrieval_1", "patch_v1"],
            )
            self.assertEqual(reviewer_request_ids, ["task_v1", "plan_v1", "patch_v1"])
            self.assertEqual(tester_request_ids, ["task_v1", "patch_v1", "review_v1"])

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
