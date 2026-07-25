from __future__ import annotations

from collections import Counter
import json
import importlib.util
import math
import os
import re
import time
from hashlib import sha256
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Protocol, Sequence, TypedDict

from swebench.inference.runtime.segment_materializer import (
    ContextSegment,
    ContextSegmentGroup,
    ResidencyTier,
    RuntimeResidencySnapshot,
    SegmentIdentity,
    SegmentRuntime,
    SegmentedGenerationRequest,
)
from swebench.inference.runtime.vllm_adapter import (
    OpenAICompatibleVLLMAdapter,
    VLLMBackendRequest,
)
from swebench.inference.trace.semantic_state import TraceLogger


PASS_TOKENS = {"PASS", "RESOLVED", "SUCCESS"}
FAIL_TOKENS = {"FAIL", "UNRESOLVED", "ERROR"}


class LangGraphRunnerState(TypedDict, total=False):
    iteration: int
    status: str
    final_plan_text: str
    final_patch_text: str
    final_verdict_text: str
    current_route_id: str | None
    current_plan_id: str | None
    current_patch_id: str | None
    current_review_id: str | None
    current_verification_id: str | None
    diagnostic_state_id: str | None
    continue_loop: bool


def _load_langgraph_symbols():
    if importlib.util.find_spec("langgraph") is None:
        raise RuntimeError(
            "langgraph package is required for the real LangGraph runner. "
            "Install it in the active interpreter, or use the existing mock-style runner."
        )
    from langgraph.graph import END, START, StateGraph

    return StateGraph, START, END


def approx_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _segment_token_sum(segments: Sequence[ContextSegment] | None) -> int:
    return sum(segment.token_count for segment in segments or ())


def _prompt_metrics(
    *,
    system_prompt: str,
    user_prompt: str,
    segment_request: SegmentedGenerationRequest | None,
    messages: Sequence[Mapping[str, str]],
    prompt_tokens_override: int | None = None,
) -> Dict[str, int]:
    system_prompt_tokens_estimate = approx_token_count(system_prompt)
    user_prompt_tokens_estimate = approx_token_count(user_prompt)
    ordered_segment_tokens = _segment_token_sum(
        segment_request.ordered_segments if segment_request is not None else ()
    )
    request_segment_tokens = _segment_token_sum(
        segment_request.request_segments if segment_request is not None else ()
    )
    assembled_prompt_tokens_estimate = sum(
        approx_token_count(str(message.get("content", ""))) for message in messages
    )
    prompt_tokens = (
        int(prompt_tokens_override)
        if prompt_tokens_override is not None
        else assembled_prompt_tokens_estimate
    )
    prompt_payload_tokens_estimate = (
        system_prompt_tokens_estimate + user_prompt_tokens_estimate + request_segment_tokens
    )
    duplicate_prompt_tokens_estimate = max(
        0, prompt_tokens - prompt_payload_tokens_estimate
    )
    return {
        "system_prompt_tokens_estimate": system_prompt_tokens_estimate,
        "user_prompt_tokens_estimate": user_prompt_tokens_estimate,
        "ordered_segment_tokens": ordered_segment_tokens,
        "request_segment_tokens": request_segment_tokens,
        "assembled_prompt_tokens_estimate": assembled_prompt_tokens_estimate,
        "prompt_tokens": prompt_tokens,
        "prompt_payload_tokens_estimate": prompt_payload_tokens_estimate,
        "duplicate_prompt_tokens_estimate": duplicate_prompt_tokens_estimate,
    }


def size_bytes(text: str) -> int:
    return max(1, len(text.encode("utf-8")))


def sanitize_state_suffix(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return sanitized or "state"


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def make_code_text(files: Mapping[str, str]) -> str:
    blocks = []
    for filename, contents in sorted(files.items()):
        blocks.append(f"[start of {filename}]\n{contents}\n[end of {filename}]")
    return "\n".join(blocks)


def module_metadata(
    *,
    context_module: str,
    lifecycle_class: str,
    is_immutable: bool,
    is_shared: bool,
    is_ephemeral: bool,
    update_cause: str,
    **extra: object,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "context_module": context_module,
        "lifecycle_class": lifecycle_class,
        "is_immutable": is_immutable,
        "is_shared": is_shared,
        "is_ephemeral": is_ephemeral,
        "update_cause": update_cause,
    }
    payload.update(extra)
    return payload


def load_instances_from_path(path: str | Path) -> List["WorkflowInstance"]:
    path = Path(path)
    if path.suffix == ".jsonl":
        raw_instances = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_instances = payload if isinstance(payload, list) else [payload]
    return [WorkflowInstance.from_mapping(item) for item in raw_instances]


@dataclass
class WorkflowInstance:
    instance_id: str
    problem_statement: str
    file_contents: Dict[str, str]
    readmes: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "WorkflowInstance":
        file_contents = dict(payload.get("file_contents", {}))
        readmes = dict(payload.get("readmes", {}))
        if not file_contents:
            raise ValueError("instance must include non-empty file_contents")
        instance_id = str(payload["instance_id"])
        problem_statement = str(payload["problem_statement"])
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"instance_id", "problem_statement", "file_contents", "readmes"}
        }
        return cls(
            instance_id=instance_id,
            problem_statement=problem_statement,
            file_contents={str(k): str(v) for k, v in file_contents.items()},
            readmes={str(k): str(v) for k, v in readmes.items()},
            metadata=metadata,
        )


def build_demo_instance() -> WorkflowInstance:
    return WorkflowInstance(
        instance_id="demo__demo-1",
        problem_statement=(
            "Fix divide-by-zero handling in `safe_ratio` and make the failing test pass."
        ),
        file_contents={
            "src/math_utils.py": (
                "def safe_ratio(a, b):\n"
                "    return a / b\n"
            ),
            "tests/test_math_utils.py": (
                "from src.math_utils import safe_ratio\n\n"
                "def test_safe_ratio_zero_divisor():\n"
                "    assert safe_ratio(10, 0) == 0\n"
            ),
        },
        readmes={
            "README.md": (
                "This project contains utility math helpers. "
                "Returning 0 for zero divisors is the intended behavior."
            )
        },
    )


def build_deep_research_demo_instance() -> WorkflowInstance:
    return WorkflowInstance(
        instance_id="deep-research__demo-1",
        problem_statement=(
            "Write a concise survey on speculative decoding for LLM inference, covering "
            "core ideas, system tradeoffs, and open implementation challenges."
        ),
        file_contents={
            "papers/speculative_decoding_overview.md": (
                "Speculative decoding accelerates generation by drafting multiple tokens with "
                "a cheap proposal model and then verifying them with a stronger target model. "
                "The main benefit depends on proposal accuracy, verification batching, and "
                "scheduler overheads. Common tradeoffs include wasted verification work, "
                "proposal-target mismatch, and memory pressure from maintaining additional "
                "model state.\n"
            ),
            "papers/system_tradeoffs.md": (
                "Serving systems must balance proposal quality, batch formation, and KV-cache "
                "reuse. Throughput gains can disappear when draft quality is low or when "
                "verification causes pipeline bubbles. Integration with prefix caching and "
                "offloading is promising, but scheduler complexity rises with heterogeneous "
                "request lengths and evolving context windows.\n"
            ),
            "papers/open_problems.md": (
                "Open problems include adaptive proposal depth, multi-tenant fairness, "
                "token-level fallback strategies, and interactions with retrieval-augmented "
                "or agentic workloads. Measuring end-to-end latency requires separating "
                "frontend orchestration costs from backend prefill and decode work.\n"
            ),
        },
        readmes={
            "README.md": (
                "Treat each markdown file as one retrieved source document. The final answer "
                "should synthesize repeated evidence, note systems implications, and preserve "
                "a stable outline across multiple planner/writer iterations."
            )
        },
    )


@dataclass(frozen=True)
class OpenDeepResearchPromptPack:
    repo_path: Path
    clarify_with_user_instructions: str
    transform_messages_into_research_topic_prompt: str
    lead_researcher_prompt: str
    research_system_prompt: str
    compress_research_system_prompt: str
    compress_research_simple_human_message: str
    final_report_generation_prompt: str


@dataclass(frozen=True)
class SWEAgentPromptPack:
    repo_path: Path
    config_path: Path
    system_template: str
    instance_template: str
    next_step_template: str
    next_step_no_output_template: str
    submit_review_messages: tuple[str, ...]


@dataclass(frozen=True)
class OpenHandsPromptPack:
    repo_path: Path
    config_path: Path
    agents_path: Path
    codeact_agent_path: Path
    default_agent_name: str
    agents_guidance: str
    development_guidance: str


def _default_open_deep_research_repo_path() -> Path | None:
    candidate = Path(__file__).resolve().parents[3] / ".external" / "open_deep_research"
    return candidate if candidate.exists() else None


def _default_swe_agent_repo_path() -> Path | None:
    candidate = Path(__file__).resolve().parents[3] / ".external" / "swe-agent"
    return candidate if candidate.exists() else None


def _default_openhands_repo_path() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[3] / ".external" / "openhands",
        Path(__file__).resolve().parents[3] / ".external" / "OpenHands",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _extract_yaml_block_scalar(source: str, key: str) -> str:
    pattern = re.compile(rf"^(?P<indent>\s*){re.escape(key)}:\s*[>|][+-]?\s*$")
    lines = source.splitlines()
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match is None:
            continue
        block_indent = len(match.group("indent")) + 2
        collected: List[str] = []
        for next_line in lines[index + 1 :]:
            stripped = next_line.strip()
            current_indent = len(next_line) - len(next_line.lstrip(" "))
            if stripped and current_indent < block_indent:
                break
            if not stripped:
                collected.append("")
            elif len(next_line) >= block_indent:
                collected.append(next_line[block_indent:])
            else:
                collected.append("")
        return "\n".join(collected).rstrip()
    raise RuntimeError(f"missing YAML block scalar for {key}")


def _extract_yaml_block_scalar_list(source: str, key: str) -> List[str]:
    pattern = re.compile(rf"^(?P<indent>\s*){re.escape(key)}:\s*$")
    item_pattern = re.compile(r"^(?P<indent>\s*)-\s*[>|][+-]?\s*$")
    lines = source.splitlines()
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match is None:
            continue
        key_indent = len(match.group("indent"))
        values: List[str] = []
        cursor = index + 1
        while cursor < len(lines):
            item_line = lines[cursor]
            stripped = item_line.strip()
            current_indent = len(item_line) - len(item_line.lstrip(" "))
            if stripped and current_indent <= key_indent:
                break
            item_match = item_pattern.match(item_line)
            if item_match is None:
                cursor += 1
                continue
            block_indent = len(item_match.group("indent")) + 2
            cursor += 1
            collected: List[str] = []
            while cursor < len(lines):
                next_line = lines[cursor]
                next_stripped = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_stripped and next_indent < block_indent:
                    break
                if not next_stripped:
                    collected.append("")
                elif len(next_line) >= block_indent:
                    collected.append(next_line[block_indent:])
                else:
                    collected.append("")
                cursor += 1
            values.append("\n".join(collected).rstrip())
        return values
    return []


def _render_template_variables(template: str, **variables: object) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
    return rendered


def _extract_toml_string_assignment(source: str, key: str) -> str | None:
    pattern = re.compile(
        rf"^\s*#?\s*{re.escape(key)}\s*=\s*\"(?P<value>[^\"]+)\"\s*$",
        re.MULTILINE,
    )
    match = pattern.search(source)
    return match.group("value") if match is not None else None


def load_open_deep_research_prompt_pack(
    repo_path: str | Path | None = None,
) -> OpenDeepResearchPromptPack:
    resolved_repo_path = (
        Path(repo_path).expanduser().resolve()
        if repo_path is not None
        else _default_open_deep_research_repo_path()
    )
    if resolved_repo_path is None or not resolved_repo_path.exists():
        raise RuntimeError(
            "open_deep_research repo not found. Clone langchain-ai/open_deep_research "
            "and pass --open_deep_research_path, or place it at "
            "'./.external/open_deep_research'."
        )

    langgraph_manifest = resolved_repo_path / "langgraph.json"
    prompts_path = (
        resolved_repo_path / "src" / "open_deep_research" / "prompts.py"
    )
    if not langgraph_manifest.exists() or not prompts_path.exists():
        raise RuntimeError(
            f"{resolved_repo_path} does not look like an open_deep_research checkout"
        )

    module_name = (
        "swebench_external_open_deep_research_prompts_"
        + sha256(str(prompts_path).encode("utf-8")).hexdigest()[:12]
    )
    spec = importlib.util.spec_from_file_location(module_name, prompts_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load prompts from {prompts_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required_fields = [
        "clarify_with_user_instructions",
        "transform_messages_into_research_topic_prompt",
        "lead_researcher_prompt",
        "research_system_prompt",
        "compress_research_system_prompt",
        "compress_research_simple_human_message",
        "final_report_generation_prompt",
    ]
    missing = [field_name for field_name in required_fields if not hasattr(module, field_name)]
    if missing:
        raise RuntimeError(
            f"open_deep_research prompts missing expected fields: {', '.join(sorted(missing))}"
        )

    return OpenDeepResearchPromptPack(
        repo_path=resolved_repo_path,
        clarify_with_user_instructions=str(module.clarify_with_user_instructions),
        transform_messages_into_research_topic_prompt=str(
            module.transform_messages_into_research_topic_prompt
        ),
        lead_researcher_prompt=str(module.lead_researcher_prompt),
        research_system_prompt=str(module.research_system_prompt),
        compress_research_system_prompt=str(module.compress_research_system_prompt),
        compress_research_simple_human_message=str(
            module.compress_research_simple_human_message
        ),
        final_report_generation_prompt=str(module.final_report_generation_prompt),
    )


def load_swe_agent_prompt_pack(
    repo_path: str | Path | None = None,
) -> SWEAgentPromptPack:
    resolved_repo_path = (
        Path(repo_path).expanduser().resolve()
        if repo_path is not None
        else _default_swe_agent_repo_path()
    )
    if resolved_repo_path is None or not resolved_repo_path.exists():
        raise RuntimeError(
            "swe-agent repo not found. Clone swe-agent/swe-agent and pass "
            "--swe_agent_path, or place it at './.external/swe-agent'."
        )

    config_path = resolved_repo_path / "config" / "default.yaml"
    if not config_path.exists():
        raise RuntimeError(
            f"{resolved_repo_path} does not look like a swe-agent checkout"
        )

    source = config_path.read_text(encoding="utf-8")
    return SWEAgentPromptPack(
        repo_path=resolved_repo_path,
        config_path=config_path,
        system_template=_extract_yaml_block_scalar(source, "system_template"),
        instance_template=_extract_yaml_block_scalar(source, "instance_template"),
        next_step_template=_extract_yaml_block_scalar(source, "next_step_template"),
        next_step_no_output_template=_extract_yaml_block_scalar(
            source, "next_step_no_output_template"
        ),
        submit_review_messages=tuple(
            _extract_yaml_block_scalar_list(source, "SUBMIT_REVIEW_MESSAGES")
        ),
    )


def load_openhands_prompt_pack(
    repo_path: str | Path | None = None,
) -> OpenHandsPromptPack:
    resolved_repo_path = (
        Path(repo_path).expanduser().resolve()
        if repo_path is not None
        else _default_openhands_repo_path()
    )
    if resolved_repo_path is None or not resolved_repo_path.exists():
        raise RuntimeError(
            "OpenHands repo not found. Clone OpenHands/openhands and pass "
            "--openhands_path, or place it at './.external/openhands'."
        )

    config_candidates = [
        resolved_repo_path / "config.template.toml",
        resolved_repo_path / "config.toml",
    ]
    config_path = next((path for path in config_candidates if path.exists()), None)
    agents_path = resolved_repo_path / "AGENTS.md"
    development_path = resolved_repo_path / "Development.md"
    codeact_candidates = [
        resolved_repo_path / "openhands" / "agenthub" / "codeact_agent" / "codeact_agent.py",
        resolved_repo_path / "openhands" / "agenthub" / "codeact_agent" / "agent.py",
    ]
    codeact_agent_path = next((path for path in codeact_candidates if path.exists()), None)
    if codeact_agent_path is None:
        codeact_agent_path = next(
            (
                path
                for path in resolved_repo_path.rglob("codeact_agent.py")
                if ".venv" not in path.parts
            ),
            None,
        )
    if config_path is None and not agents_path.exists() and not development_path.exists():
        raise RuntimeError(
            f"{resolved_repo_path} does not look like an OpenHands/openhands checkout"
        )
    development_guidance = (
        development_path.read_text(encoding="utf-8")
        if development_path.exists()
        else ""
    )
    config_source = (
        config_path.read_text(encoding="utf-8")
        if config_path is not None
        else ""
    )
    default_agent_name = (
        _extract_toml_string_assignment(config_source, "default_agent")
        or "CodeActAgent"
    )
    if codeact_agent_path is None:
        codeact_agent_path = resolved_repo_path
    return OpenHandsPromptPack(
        repo_path=resolved_repo_path,
        config_path=config_path if config_path is not None else resolved_repo_path,
        agents_path=agents_path if agents_path.exists() else resolved_repo_path,
        codeact_agent_path=codeact_agent_path,
        default_agent_name=default_agent_name,
        agents_guidance=(
            agents_path.read_text(encoding="utf-8")
            if agents_path.exists()
            else "OpenHands repository guidance file AGENTS.md was not present in this checkout."
        ),
        development_guidance=development_guidance,
    )


class ModelBackend(Protocol):
    name: str

    def complete(
        self,
        *,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        iteration: int,
        instance: WorkflowInstance,
        prompt_mode: str = "monolithic",
        prompt_group: ContextSegmentGroup | None = None,
        segment_request: SegmentedGenerationRequest | None = None,
        materializer_snapshot: RuntimeResidencySnapshot | None = None,
    ) -> str:
        ...

    @property
    def runtime_model_id(self) -> str:
        ...

    @property
    def runtime_tokenizer_id(self) -> str:
        ...

    def runtime_inference_config(
        self,
        *,
        step_name: str,
        prompt_mode: str,
    ) -> Mapping[str, object]:
        ...

    def drain_call_records(self) -> List[Dict[str, object]]:
        ...


class StubModelBackend:
    name = "stub"

    def __init__(self) -> None:
        self._call_records: List[Dict[str, object]] = []

    def complete(
        self,
        *,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        iteration: int,
        instance: WorkflowInstance,
        prompt_mode: str = "monolithic",
        prompt_group: ContextSegmentGroup | None = None,
        segment_request: SegmentedGenerationRequest | None = None,
        materializer_snapshot: RuntimeResidencySnapshot | None = None,
    ) -> str:
        call_started_at = time.perf_counter()
        message_build_started_at = time.perf_counter()
        messages = (
            segment_request.to_openai_messages(fallback_system_prompt=system_prompt)
            if segment_request is not None
            else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        frontend_message_build_ms = (
            time.perf_counter() - message_build_started_at
        ) * 1000.0
        prompt_metrics_started_at = time.perf_counter()
        prompt_metrics = _prompt_metrics(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            segment_request=segment_request,
            messages=messages,
        )
        frontend_token_estimate_ms = (
            time.perf_counter() - prompt_metrics_started_at
        ) * 1000.0
        if step_name == "planner":
            text = (
                f"Plan v{iteration}:\n"
                "1. Inspect `safe_ratio` and the failing test.\n"
                "2. Add zero-divisor handling.\n"
                "3. Re-run verification.\n"
            )
        elif step_name == "router":
            text = (
                f"Route v{iteration}: plan_then_patch\n"
                "Reason: retrieved evidence suggests a localized source fix plus lightweight verification.\n"
            )
        elif step_name == "coder":
            text = (
                "--- a/src/math_utils.py\n"
                "+++ b/src/math_utils.py\n"
                "@@ -1,2 +1,4 @@\n"
                " def safe_ratio(a, b):\n"
                "-    return a / b\n"
                "+    if b == 0:\n"
                "+        return 0\n"
                "+    return a / b\n"
            )
        elif step_name == "tester":
            if iteration == 1:
                text = (
                    "FAIL\n"
                    "The patch handles zero divisors, but you should verify edge cases and restate the intended invariant."
                )
            else:
                text = "PASS\nThe patch matches the stated requirement and the targeted test should pass."
        elif step_name == "reviewer":
            if iteration == 1:
                text = (
                    "Review v1:\n"
                    "- Guarding the zero divisor looks correct.\n"
                    "- Verification should explicitly confirm the zero-divisor contract.\n"
                )
            else:
                text = (
                "Review v2:\n"
                "- Patch matches the stated invariant.\n"
                "- No additional changes required before verification.\n"
                )
        elif step_name == "research_planner":
            text = (
                f"Research Plan v{iteration}:\n"
                "1. Reframe the survey question around systems tradeoffs.\n"
                "2. Prioritize evidence about proposal accuracy, verification cost, and scheduler overhead.\n"
                "3. Update the outline to separate mechanisms, systems implications, and open problems.\n"
            )
        elif step_name == "research_reader":
            text = (
                f"Reading Note v{iteration}:\n"
                "- Proposal quality drives the realized speedup.\n"
                "- Verification batching and KV reuse determine backend efficiency.\n"
                "- Frontend orchestration overhead can hide backend gains in agentic settings.\n"
            )
        elif step_name == "research_writer":
            text = (
                f"Survey Draft v{iteration}:\n"
                "Speculative decoding speeds up inference by drafting tokens with a cheap proposer "
                "and verifying them against a stronger target model. In practice, observed gains "
                "depend on proposal accuracy, batched verification, and scheduler efficiency. "
                "Systems challenges include balancing KV reuse, avoiding verification bubbles, "
                "and separating frontend request overhead from backend prefill savings.\n"
            )
        elif step_name == "research_critic":
            text = (
                f"Critique v{iteration}:\n"
                "- Clarify that not all latency savings come from model-side compute.\n"
                "- Strengthen the connection between evidence reuse and serving-layer compatibility costs.\n"
                "- Keep the final draft explicit about open runtime questions.\n"
            )
        else:
            raise ValueError(f"unknown step_name {step_name!r}")

        completion_tokens = approx_token_count(text)
        total_duration_ms = (time.perf_counter() - call_started_at) * 1000.0
        self._call_records.append(
            {
                "step_name": step_name,
                "prompt_mode": prompt_mode,
                "duration_ms": total_duration_ms,
                "backend_roundtrip_ms": 0.0,
                "frontend_message_build_ms": frontend_message_build_ms,
                "frontend_token_estimate_ms": frontend_token_estimate_ms,
                "frontend_overhead_ms": (
                    frontend_message_build_ms + frontend_token_estimate_ms
                ),
                "prompt_tokens": prompt_metrics["prompt_tokens"],
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_metrics["prompt_tokens"] + completion_tokens,
                "segment_group_id": prompt_group.group_id if prompt_group is not None else None,
                "segment_count": (
                    len(segment_request.ordered_segments) if segment_request is not None else 0
                ),
                "ordered_segment_count": (
                    len(segment_request.ordered_segments) if segment_request is not None else 0
                ),
                "request_segment_count": (
                    len(segment_request.request_segments) if segment_request is not None else 0
                ),
                "system_prompt_tokens_estimate": prompt_metrics["system_prompt_tokens_estimate"],
                "user_prompt_tokens_estimate": prompt_metrics["user_prompt_tokens_estimate"],
                "ordered_segment_tokens": prompt_metrics["ordered_segment_tokens"],
                "request_segment_tokens": prompt_metrics["request_segment_tokens"],
                "assembled_prompt_tokens_estimate": prompt_metrics[
                    "assembled_prompt_tokens_estimate"
                ],
                "prompt_payload_tokens_estimate": prompt_metrics[
                    "prompt_payload_tokens_estimate"
                ],
                "duplicate_prompt_tokens_estimate": prompt_metrics[
                    "duplicate_prompt_tokens_estimate"
                ],
                "runtime_hbm_bytes": (
                    materializer_snapshot.hbm_bytes
                    if materializer_snapshot is not None
                    else None
                ),
                "frontend_cache_hit": False,
            }
        )
        return text

    @property
    def runtime_model_id(self) -> str:
        return "stub"

    @property
    def runtime_tokenizer_id(self) -> str:
        return "stub"

    def runtime_inference_config(
        self,
        *,
        step_name: str,
        prompt_mode: str,
    ) -> Mapping[str, object]:
        return {
            "provider": self.name,
            "step_name": step_name,
            "prompt_mode": prompt_mode,
            "temperature": 0.0,
        }

    def drain_call_records(self) -> List[Dict[str, object]]:
        records = list(self._call_records)
        self._call_records.clear()
        return records


class OpenAICompatibleChatBackend:
    name = "openai-compatible"

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.name = provider_name
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_records: List[Dict[str, object]] = []

    def _client(self):
        try:
            import os
            import openai
        except ImportError as exc:
            raise RuntimeError(
                f"openai package is required for provider={self.name}"
            ) from exc

        client_kwargs = {}
        if self.base_url is not None:
            client_kwargs["base_url"] = self.base_url
        api_key = self.api_key if self.api_key is not None else os.environ.get(self.api_key_env)
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        client_kwargs["timeout"] = self.timeout
        client_kwargs["max_retries"] = self.max_retries

        if hasattr(openai, "OpenAI"):
            return openai.OpenAI(**client_kwargs), "client"
        if "base_url" in client_kwargs:
            openai.base_url = client_kwargs["base_url"]
        if "api_key" in client_kwargs:
            openai.api_key = client_kwargs["api_key"]
        if hasattr(openai, "timeout"):
            openai.timeout = client_kwargs["timeout"]
        return openai, "module"

    def complete(
        self,
        *,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        iteration: int,
        instance: WorkflowInstance,
        prompt_mode: str = "monolithic",
        prompt_group: ContextSegmentGroup | None = None,
        segment_request: SegmentedGenerationRequest | None = None,
        materializer_snapshot: RuntimeResidencySnapshot | None = None,
    ) -> str:
        client, mode = self._client()
        messages = (
            segment_request.to_openai_messages(fallback_system_prompt=system_prompt)
            if segment_request is not None
            else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        if mode == "client":
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    @property
    def runtime_model_id(self) -> str:
        return self.model

    @property
    def runtime_tokenizer_id(self) -> str:
        return self.model

    def runtime_inference_config(
        self,
        *,
        step_name: str,
        prompt_mode: str,
    ) -> Mapping[str, object]:
        return {
            "provider": self.name,
            "step_name": step_name,
            "prompt_mode": prompt_mode,
            "temperature": self.temperature,
        }

    def drain_call_records(self) -> List[Dict[str, object]]:
        records = list(self._call_records)
        self._call_records.clear()
        return records


class VLLMServerChatBackend:
    name = "vllm"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._call_records: List[Dict[str, object]] = []
        self._message_cache: Dict[str, tuple[tuple[str, str], ...]] = {}
        self.adapter = OpenAICompatibleVLLMAdapter(
            base_url=base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        *,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        iteration: int,
        instance: WorkflowInstance,
        prompt_mode: str = "monolithic",
        prompt_group: ContextSegmentGroup | None = None,
        segment_request: SegmentedGenerationRequest | None = None,
        materializer_snapshot: RuntimeResidencySnapshot | None = None,
    ) -> str:
        call_started_at = time.perf_counter()
        request = VLLMBackendRequest(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            prompt_mode=prompt_mode,
            segment_request=segment_request,
            extra_body={
                "agent_step": step_name,
                "segment_group_id": prompt_group.group_id if prompt_group is not None else None,
                "segment_count": (
                    len(segment_request.ordered_segments)
                    if segment_request is not None
                    else 0
                ),
                "runtime_hbm_bytes": (
                    materializer_snapshot.hbm_bytes
                    if materializer_snapshot is not None
                    else None
                ),
            },
        )
        request_key = sha256(
            json.dumps(
                {
                    "model": request.model,
                    "system_prompt": request.system_prompt,
                    "user_prompt": request.user_prompt,
                    "prompt_mode": request.prompt_mode,
                    "segment_group_id": (
                        prompt_group.group_id if prompt_group is not None else None
                    ),
                    "segment_request": (
                        [segment.segment_id for segment in segment_request.ordered_segments]
                        if segment_request is not None
                        else None
                    ),
                    "request_segment_ids": (
                        [segment.segment_id for segment in segment_request.request_segments]
                        if segment_request is not None
                        else None
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        message_build_started_at = time.perf_counter()
        cached_messages = self._message_cache.get(request_key)
        if cached_messages is None:
            cached_messages = tuple(
                (message["role"], message["content"]) for message in request.to_messages()
            )
            self._message_cache[request_key] = cached_messages
        request = VLLMBackendRequest(
            model=request.model,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            prompt_mode=request.prompt_mode,
            segment_request=request.segment_request,
            extra_body=request.extra_body,
            messages_override=cached_messages,
        )
        frontend_message_build_ms = (
            time.perf_counter() - message_build_started_at
        ) * 1000.0
        result = self.adapter.complete_with_details(request)
        prompt_metrics_started_at = time.perf_counter()
        prompt_metrics = _prompt_metrics(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            segment_request=segment_request,
            messages=[
                {"role": role, "content": content} for role, content in cached_messages
            ],
            prompt_tokens_override=result.prompt_tokens,
        )
        frontend_token_estimate_ms = (
            time.perf_counter() - prompt_metrics_started_at
        ) * 1000.0
        total_duration_ms = (time.perf_counter() - call_started_at) * 1000.0
        self._call_records.append(
            {
                "step_name": step_name,
                "prompt_mode": prompt_mode,
                "duration_ms": total_duration_ms,
                "backend_roundtrip_ms": result.duration_ms,
                "frontend_message_build_ms": frontend_message_build_ms,
                "frontend_token_estimate_ms": frontend_token_estimate_ms,
                "frontend_overhead_ms": (
                    frontend_message_build_ms + frontend_token_estimate_ms
                ),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "message_digest": result.message_digest,
                "segment_group_id": prompt_group.group_id if prompt_group is not None else None,
                "segment_count": (
                    len(segment_request.ordered_segments)
                    if segment_request is not None
                    else 0
                ),
                "ordered_segment_count": (
                    len(segment_request.ordered_segments)
                    if segment_request is not None
                    else 0
                ),
                "request_segment_count": (
                    len(segment_request.request_segments)
                    if segment_request is not None
                    else 0
                ),
                "system_prompt_tokens_estimate": prompt_metrics["system_prompt_tokens_estimate"],
                "user_prompt_tokens_estimate": prompt_metrics["user_prompt_tokens_estimate"],
                "ordered_segment_tokens": prompt_metrics["ordered_segment_tokens"],
                "request_segment_tokens": prompt_metrics["request_segment_tokens"],
                "assembled_prompt_tokens_estimate": prompt_metrics[
                    "assembled_prompt_tokens_estimate"
                ],
                "prompt_payload_tokens_estimate": prompt_metrics[
                    "prompt_payload_tokens_estimate"
                ],
                "duplicate_prompt_tokens_estimate": prompt_metrics[
                    "duplicate_prompt_tokens_estimate"
                ],
                "runtime_hbm_bytes": (
                    materializer_snapshot.hbm_bytes
                    if materializer_snapshot is not None
                    else None
                ),
                "frontend_cache_hit": result.frontend_cache_hit,
                "assembled_message_count": len(cached_messages),
                "request_key": request_key,
            }
        )
        return result.text

    @property
    def runtime_model_id(self) -> str:
        return self.model

    @property
    def runtime_tokenizer_id(self) -> str:
        return self.model

    def runtime_inference_config(
        self,
        *,
        step_name: str,
        prompt_mode: str,
    ) -> Mapping[str, object]:
        return {
            "provider": self.name,
            "step_name": step_name,
            "prompt_mode": prompt_mode,
            "temperature": self.temperature,
            "transport": "openai-compatible-vllm",
            "apc_enabled": True,
        }

    def drain_call_records(self) -> List[Dict[str, object]]:
        records = list(self._call_records)
        self._call_records.clear()
        return records


class OpenAIChatBackend(OpenAICompatibleChatBackend):
    name = "openai"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            provider_name="openai",
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )


class OllamaChatBackend(OpenAICompatibleChatBackend):
    name = "ollama"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        import os

        super().__init__(
            provider_name="ollama",
            model=model,
            temperature=temperature,
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",
            timeout=timeout,
            max_retries=max_retries,
        )


class GroqChatBackend(OpenAICompatibleChatBackend):
    name = "groq"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        import os

        super().__init__(
            provider_name="groq",
            model=model,
            temperature=temperature,
            base_url=base_url or os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key_env="GROQ_API_KEY",
            timeout=timeout,
            max_retries=max_retries,
        )


class AnthropicMessagesBackend:
    name = "anthropic"

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self._call_records: List[Dict[str, object]] = []

    def complete(
        self,
        *,
        step_name: str,
        system_prompt: str,
        user_prompt: str,
        iteration: int,
        instance: WorkflowInstance,
        prompt_mode: str = "monolithic",
        prompt_group: ContextSegmentGroup | None = None,
        segment_request: SegmentedGenerationRequest | None = None,
        materializer_snapshot: RuntimeResidencySnapshot | None = None,
    ) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package is required for provider=anthropic") from exc

        client = Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        return "\n".join(text_blocks)

    @property
    def runtime_model_id(self) -> str:
        return self.model

    @property
    def runtime_tokenizer_id(self) -> str:
        return self.model

    def runtime_inference_config(
        self,
        *,
        step_name: str,
        prompt_mode: str,
    ) -> Mapping[str, object]:
        return {
            "provider": self.name,
            "step_name": step_name,
            "prompt_mode": prompt_mode,
            "temperature": self.temperature,
            "max_tokens": 1024,
        }

    def drain_call_records(self) -> List[Dict[str, object]]:
        records = list(self._call_records)
        self._call_records.clear()
        return records


def make_backend(
    provider: str,
    model: str | None,
    *,
    timeout: float = 120.0,
    max_retries: int = 2,
) -> ModelBackend:
    if provider == "stub":
        return StubModelBackend()
    if provider == "openai":
        if not model:
            raise ValueError("--model is required for provider=openai")
        return OpenAIChatBackend(model=model, timeout=timeout, max_retries=max_retries)
    if provider == "ollama":
        if not model:
            raise ValueError("--model is required for provider=ollama")
        return OllamaChatBackend(model=model, timeout=timeout, max_retries=max_retries)
    if provider == "groq":
        if not model:
            raise ValueError("--model is required for provider=groq")
        return GroqChatBackend(model=model, timeout=timeout, max_retries=max_retries)
    if provider == "vllm":
        if not model:
            raise ValueError("--model is required for provider=vllm")
        return VLLMServerChatBackend(model=model, timeout=timeout, max_retries=max_retries)
    if provider == "anthropic":
        if not model:
            raise ValueError("--model is required for provider=anthropic")
        return AnthropicMessagesBackend(model=model)
    raise ValueError(f"unsupported provider {provider!r}")


@dataclass
class TraceValidationReport:
    errors: List[str]
    summary: Dict[str, object]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def load_trace_events(path: str | Path) -> List[Dict[str, object]]:
    path = Path(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_trace_events(events: Sequence[Mapping[str, object]]) -> TraceValidationReport:
    errors: List[str] = []
    created_state_ids = set()
    seen_supersede_pairs = set()
    op_counts: Dict[str, int] = {}
    state_type_counts: Dict[str, int] = {}
    last_ts = -1
    read_prompt_count = 0

    for index, event in enumerate(events):
        op = str(event.get("op"))
        op_counts[op] = op_counts.get(op, 0) + 1
        ts = int(event.get("ts", -1))
        if ts < last_ts:
            errors.append(f"event {index}: timestamps are not non-decreasing")
        last_ts = ts

        if op in {"CREATE", "DERIVE"}:
            state_id = str(event["state_id"])
            created_state_ids.add(state_id)
            state_type = str(event["state_type"])
            state_type_counts[state_type] = state_type_counts.get(state_type, 0) + 1
            supersedes = event.get("supersedes")
            if supersedes is not None and str(supersedes) not in created_state_ids:
                errors.append(f"event {index}: supersedes unknown state_id {supersedes!r}")
        elif op in {"READ", "RELEASE", "SHARE", "MATERIALIZE", "EVICT", "RELOAD"}:
            state_id = str(event["state_id"])
            if state_id not in created_state_ids:
                errors.append(f"event {index}: {op} references unknown state_id {state_id!r}")
            if op == "READ" and isinstance(event.get("metadata"), Mapping):
                metadata = event["metadata"]
                if "prompt_id" in metadata:
                    read_prompt_count += 1
        elif op == "SUPERSEDE":
            old_state_id = str(event["old_state_id"])
            new_state_id = str(event["new_state_id"])
            if old_state_id not in created_state_ids:
                errors.append(f"event {index}: SUPERSEDE old_state_id unknown {old_state_id!r}")
            if new_state_id not in created_state_ids:
                errors.append(f"event {index}: SUPERSEDE new_state_id unknown {new_state_id!r}")
            seen_supersede_pairs.add((old_state_id, new_state_id))

    for event in events:
        if event.get("op") in {"CREATE", "DERIVE"} and event.get("supersedes") is not None:
            pair = (str(event["supersedes"]), str(event["state_id"]))
            if pair not in seen_supersede_pairs:
                errors.append(
                    f"missing SUPERSEDE event for derived state {pair[1]!r} superseding {pair[0]!r}"
                )

    summary = {
        "event_count": len(events),
        "op_counts": op_counts,
        "state_type_counts": state_type_counts,
        "read_prompt_count": read_prompt_count,
    }
    if op_counts.get("READ", 0) == 0:
        errors.append("trace contains no READ events")
    if read_prompt_count == 0:
        errors.append("trace contains no prompt-tagged READ events")
    return TraceValidationReport(errors=errors, summary=summary)


class TracedAgentRunner:
    def __init__(
        self,
        *,
        backend: ModelBackend,
        trace_dir: str | Path,
        tenant_id: str = "local",
        max_iterations: int = 2,
        max_files: int = 5,
        prompt_runtime_mode: str = "monolithic",
    ) -> None:
        self.backend = backend
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.tenant_id = tenant_id
        self.max_iterations = max(1, max_iterations)
        self.max_files = max(1, max_files)
        if prompt_runtime_mode not in {"monolithic", "segment_aware"}:
            raise ValueError(
                "prompt_runtime_mode must be one of {'monolithic', 'segment_aware'}"
            )
        self.prompt_runtime_mode = prompt_runtime_mode
        self._reset_monolithic_prompt_runtime_state()
        self._reclaim_grace_by_role = {
            "scratch": 0.0,
            "evidence": 5.0,
            "plan": 5.0,
            "system": 3600.0,
            "task": 3600.0,
        }

    def _reset_monolithic_prompt_runtime_state(self) -> None:
        self._monolithic_prompt_versions_by_consumer: Dict[str, int] = {}
        self._monolithic_prompt_current_by_consumer: Dict[str, Dict[str, object]] = {}
        self._runtime_segment_cache: Dict[str, ContextSegment] = {}

    def _tier_from_materialization(self, materialization: str | None) -> ResidencyTier | None:
        if materialization == "HBM":
            return ResidencyTier.HBM
        if materialization == "CPU":
            return ResidencyTier.CPU
        if materialization in {"DISK", "NONE"}:
            return ResidencyTier.EVICTED
        return None

    def _register_runtime_segment(
        self,
        *,
        materializer: SegmentRuntime,
        handle,
        workflow_id: str,
        module: str,
        role: str,
        text: str,
        materialization: str | None,
        is_shared: bool,
        is_immutable: bool,
        is_ephemeral: bool,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        segment = ContextSegment(
            state_id=handle.state_id,
            identity=SegmentIdentity(logical_key=handle.logical_key, module=module),
            version=handle.version,
            size_bytes=size_bytes(text),
            token_count=approx_token_count(text),
            workflow_id=workflow_id,
            text=text,
            recompute_cost=max(1.0, approx_token_count(text) / 16.0),
            reload_cost=max(0.5, size_bytes(text) / 256.0),
            is_shared=is_shared,
            is_immutable=is_immutable,
            is_ephemeral=is_ephemeral,
            metadata={"role": role, **dict(metadata or {})},
            residency_hint=self._tier_from_materialization(materialization),
        )
        self._runtime_segment_cache[segment.state_id] = segment
        if self.prompt_runtime_mode != "segment_aware":
            return
        materializer.register_segment(segment)

    def _release_runtime_segment(
        self,
        *,
        materializer: SegmentRuntime,
        state_id: str | None,
    ) -> None:
        if self.prompt_runtime_mode != "segment_aware":
            return
        if state_id is None:
            return
        materializer.release_segment(state_id)

    def _request_segment_ids_for_step(
        self,
        *,
        step_name: str,
        segments: Sequence[Mapping[str, object]],
    ) -> List[str]:
        request_segment_ids: List[str] = []
        for segment in segments:
            state_id = str(segment["state_id"])
            role = str(segment.get("segment_role", "")).lower()
            include = True
            if role in {"system", "task"}:
                include = False
            elif step_name == "coder" and role in {"plan", "artifact"}:
                include = False
            elif step_name == "coder" and role == "evidence" and state_id.startswith("retrieval_"):
                include = False
            elif step_name == "reviewer" and role in {"plan", "artifact"}:
                include = False
            elif step_name == "tester" and role in {"plan", "artifact"}:
                include = False
            if include:
                request_segment_ids.append(state_id)
        return request_segment_ids

    def _prepare_monolithic_prompt(
        self,
        *,
        materializer: SegmentRuntime,
        trace: TraceLogger,
        workflow_id: str,
        consumer: str,
        step_name: str,
        prompt_id: str,
        system_prompt: str,
        user_prompt: str,
        segments: Sequence[Mapping[str, object]],
        request_segment_ids: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[
        ContextSegmentGroup,
        SegmentedGenerationRequest,
        RuntimeResidencySnapshot,
    ]:
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_signature = sha256(
            json.dumps(prompt_messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        current = self._monolithic_prompt_current_by_consumer.get(consumer)
        if current is None or current["signature"] != prompt_signature:
            version = self._monolithic_prompt_versions_by_consumer.get(consumer, 0) + 1
            self._monolithic_prompt_versions_by_consumer[consumer] = version
            state_id = f"mono_{sanitize_state_suffix(consumer)}_v{version}"
            segment = ContextSegment(
                state_id=state_id,
                identity=SegmentIdentity(
                    logical_key=f"runtime/monolithic/{consumer}",
                    module="monolithic_context",
                ),
                version=version,
                role="monolithic",
                size_bytes=size_bytes(system_prompt) + size_bytes(user_prompt),
                token_count=approx_token_count(system_prompt) + approx_token_count(user_prompt),
                workflow_id=workflow_id,
                text=f"{system_prompt}\n\n{user_prompt}",
                recompute_cost=max(
                    1.0,
                    (
                        approx_token_count(system_prompt) + approx_token_count(user_prompt)
                    )
                    / 16.0,
                ),
                reload_cost=max(
                    0.5,
                    (size_bytes(system_prompt) + size_bytes(user_prompt)) / 256.0,
                ),
                is_shared=False,
                is_immutable=False,
                is_ephemeral=False,
                metadata={
                    "role": "monolithic",
                    "consumer": consumer,
                    "prompt_id": prompt_id,
                    "step_name": step_name,
                    "signature": prompt_signature,
                },
                residency_hint=ResidencyTier.HBM,
            )
            previous_state_id = (
                str(current["state_id"]) if current is not None else None
            )
            if previous_state_id is None:
                materializer.register_segment(segment)
            else:
                materializer.supersede_segment(previous_state_id, segment)
            self._monolithic_prompt_current_by_consumer[consumer] = {
                "state_id": state_id,
                "signature": prompt_signature,
            }

        state_id = str(self._monolithic_prompt_current_by_consumer[consumer]["state_id"])
        group = materializer.build_group(
            group_id=prompt_id,
            workflow_id=workflow_id,
            consumer=consumer,
            ordered_segment_ids=[state_id],
            prompt_id=prompt_id,
        )
        materializer.prepare_group_read(
            group,
            model_id=self.backend.runtime_model_id,
            tokenizer_id=self.backend.runtime_tokenizer_id,
            inference_config=self.backend.runtime_inference_config(
                step_name=step_name,
                prompt_mode="monolithic",
            ),
        )
        snapshot = materializer.snapshot()
        flattened_ordered_segments = tuple(
            self._runtime_segment_cache[str(segment["state_id"])] for segment in segments
        )
        flattened_request_segments = tuple(
            self._runtime_segment_cache[state_id]
            for state_id in (
                tuple(request_segment_ids)
                if request_segment_ids is not None
                else tuple(str(segment["state_id"]) for segment in segments)
            )
        )
        trace.log_prompt_segments(
            consumer=consumer,
            prompt_id=prompt_id,
            segments=segments,
            metadata={
                **dict(metadata or {}),
                "runtime_group_id": group.group_id,
                "runtime_prompt_mode": "monolithic",
                "runtime_model_id": self.backend.runtime_model_id,
                "runtime_tokenizer_id": self.backend.runtime_tokenizer_id,
                "runtime_hbm_bytes": snapshot.hbm_bytes,
                "runtime_cpu_bytes": snapshot.cpu_bytes,
                "runtime_monolithic_state_id": state_id,
            },
        )
        return (
            group,
            SegmentedGenerationRequest(
                ordered_segments=flattened_ordered_segments,
                request_segments=flattened_request_segments,
                fallback_system_prompt=system_prompt,
                fallback_user_prompt=user_prompt,
                enable_prefix_caching=False,
            ),
            snapshot,
        )

    def _release_monolithic_prompt_runtime(
        self,
        *,
        materializer: SegmentRuntime,
    ) -> None:
        for current in list(self._monolithic_prompt_current_by_consumer.values()):
            materializer.release_segment(str(current["state_id"]))
        self._reset_monolithic_prompt_runtime_state()

    def _prepare_segmented_prompt(
        self,
        *,
        materializer: SegmentRuntime,
        trace: TraceLogger,
        workflow_id: str,
        consumer: str,
        step_name: str,
        prompt_id: str,
        system_prompt: str = "",
        user_prompt: str = "",
        segments: Sequence[Mapping[str, object]],
        request_segment_ids: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[ContextSegmentGroup, SegmentedGenerationRequest, RuntimeResidencySnapshot]:
        group = materializer.build_group(
            group_id=prompt_id,
            workflow_id=workflow_id,
            consumer=consumer,
            ordered_segment_ids=[str(segment["state_id"]) for segment in segments],
            prompt_id=prompt_id,
            request_segment_ids=request_segment_ids,
        )
        materializer.prepare_group_read(
            group,
            model_id=self.backend.runtime_model_id,
            tokenizer_id=self.backend.runtime_tokenizer_id,
            inference_config=self.backend.runtime_inference_config(
                step_name=step_name,
                prompt_mode="segment_aware",
            ),
        )
        snapshot = materializer.snapshot()
        trace.log_prompt_segments(
            consumer=consumer,
            prompt_id=prompt_id,
            segments=segments,
            metadata={
                **dict(metadata or {}),
                "runtime_group_id": group.group_id,
                "runtime_prompt_mode": "segment_aware",
                "runtime_model_id": self.backend.runtime_model_id,
                "runtime_tokenizer_id": self.backend.runtime_tokenizer_id,
                "runtime_hbm_bytes": snapshot.hbm_bytes,
                "runtime_cpu_bytes": snapshot.cpu_bytes,
                "runtime_request_segment_count": len(group.request_segment_ids),
            },
        )
        return (
            group,
            materializer.build_vllm_request(
                group,
                fallback_system_prompt=system_prompt,
                fallback_user_prompt=user_prompt,
            ),
            snapshot,
        )

    def _prepare_langgraph_prompt_runtime(
        self,
        *,
        materializer: SegmentRuntime,
        trace: TraceLogger,
        workflow_id: str,
        consumer: str,
        step_name: str,
        prompt_id: str,
        system_prompt: str,
        user_prompt: str,
        segments: Sequence[Mapping[str, object]],
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[
        str,
        ContextSegmentGroup | None,
        SegmentedGenerationRequest | None,
        RuntimeResidencySnapshot | None,
    ]:
        if self.prompt_runtime_mode == "segment_aware":
            group, request, snapshot = self._prepare_segmented_prompt(
                materializer=materializer,
                trace=trace,
                workflow_id=workflow_id,
                consumer=consumer,
                step_name=step_name,
                prompt_id=prompt_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                segments=segments,
                request_segment_ids=self._request_segment_ids_for_step(
                    step_name=step_name,
                    segments=segments,
                ),
                metadata=metadata,
            )
            return "segment_aware", group, request, snapshot

        request_segment_ids = self._request_segment_ids_for_step(
            step_name=step_name,
            segments=segments,
        )
        group, request, snapshot = self._prepare_monolithic_prompt(
            materializer=materializer,
            trace=trace,
            workflow_id=workflow_id,
            consumer=consumer,
            step_name=step_name,
            prompt_id=prompt_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            segments=segments,
            request_segment_ids=request_segment_ids,
            metadata=metadata,
        )
        return "monolithic", group, request, snapshot

    def _runtime_event_summary(self, materializer: SegmentRuntime) -> Dict[str, object]:
        events = materializer.events()
        op_counts = Counter(event.operation for event in events)
        lookup_status_counts = Counter(
            event.lookup_status for event in events if event.lookup_status is not None
        )
        return {
            "event_count": len(events),
            "op_counts": dict(sorted(op_counts.items())),
            "lookup_status_counts": dict(sorted(lookup_status_counts.items())),
        }

    def _write_runtime_events(
        self,
        *,
        materializer: SegmentRuntime,
        runtime_event_path: Path,
        workflow_id: str,
        agent_family: str,
    ) -> None:
        runtime_event_path.parent.mkdir(parents=True, exist_ok=True)
        with runtime_event_path.open("w", encoding="utf-8") as handle:
            for event in materializer.events():
                segment = materializer.get_segment(event.segment_id)
                record = {
                    "workflow_id": workflow_id,
                    "segment_id": event.segment_id,
                    "logical_id": event.logical_id,
                    "version": event.version,
                    "role": str(segment.role.value if hasattr(segment.role, "value") else segment.role),
                    "module": segment.identity.module,
                    "operation": event.operation,
                    "timestamp": event.timestamp,
                    "from_semantic_state": event.from_semantic_state,
                    "to_semantic_state": event.to_semantic_state,
                    "from_residency_state": event.from_residency_state,
                    "to_residency_state": event.to_residency_state,
                    "lookup_status": event.lookup_status,
                    "execution_context_digest": event.execution_context_digest,
                    "reason": event.reason,
                    "size_bytes": event.size_bytes,
                    "token_count": segment.token_count,
                    "agent_family": agent_family,
                    "provider": getattr(self.backend, "name", self.backend.__class__.__name__),
                }
                handle.write(json.dumps(record) + "\n")

    def _backend_call_summary(
        self,
        call_records: Sequence[Mapping[str, object]],
    ) -> Dict[str, object]:
        if not call_records:
            return {
                "request_count": 0,
                "total_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
                "total_backend_roundtrip_ms": 0.0,
                "avg_backend_roundtrip_ms": 0.0,
                "total_frontend_message_build_ms": 0.0,
                "avg_frontend_message_build_ms": 0.0,
                "total_frontend_token_estimate_ms": 0.0,
                "avg_frontend_token_estimate_ms": 0.0,
                "total_frontend_overhead_ms": 0.0,
                "avg_frontend_overhead_ms": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "avg_prompt_tokens": 0.0,
                "duration_ms_per_1k_prompt_tokens": 0.0,
                "frontend_cache_hits": 0,
                "frontend_cache_hit_rate": 0.0,
                "total_system_prompt_tokens_estimate": 0,
                "total_user_prompt_tokens_estimate": 0,
                "total_ordered_segment_tokens": 0,
                "total_request_segment_tokens": 0,
                "total_assembled_prompt_tokens_estimate": 0,
                "total_prompt_payload_tokens_estimate": 0,
                "total_duplicate_prompt_tokens_estimate": 0,
                "step_rows": [],
            }

        total_duration_ms = sum(float(record.get("duration_ms", 0.0)) for record in call_records)
        total_backend_roundtrip_ms = sum(
            float(record.get("backend_roundtrip_ms", 0.0)) for record in call_records
        )
        total_frontend_message_build_ms = sum(
            float(record.get("frontend_message_build_ms", 0.0)) for record in call_records
        )
        total_frontend_token_estimate_ms = sum(
            float(record.get("frontend_token_estimate_ms", 0.0)) for record in call_records
        )
        total_frontend_overhead_ms = sum(
            float(record.get("frontend_overhead_ms", 0.0)) for record in call_records
        )
        total_prompt_tokens = sum(int(record.get("prompt_tokens") or 0) for record in call_records)
        total_completion_tokens = sum(
            int(record.get("completion_tokens") or 0) for record in call_records
        )
        total_tokens = sum(int(record.get("total_tokens") or 0) for record in call_records)
        frontend_cache_hits = sum(
            1 for record in call_records if bool(record.get("frontend_cache_hit", False))
        )
        total_system_prompt_tokens_estimate = sum(
            int(record.get("system_prompt_tokens_estimate") or 0) for record in call_records
        )
        total_user_prompt_tokens_estimate = sum(
            int(record.get("user_prompt_tokens_estimate") or 0) for record in call_records
        )
        total_ordered_segment_tokens = sum(
            int(record.get("ordered_segment_tokens") or 0) for record in call_records
        )
        total_request_segment_tokens = sum(
            int(record.get("request_segment_tokens") or 0) for record in call_records
        )
        total_assembled_prompt_tokens_estimate = sum(
            int(record.get("assembled_prompt_tokens_estimate") or 0) for record in call_records
        )
        total_prompt_payload_tokens_estimate = sum(
            int(record.get("prompt_payload_tokens_estimate") or 0) for record in call_records
        )
        total_duplicate_prompt_tokens_estimate = sum(
            int(record.get("duplicate_prompt_tokens_estimate") or 0) for record in call_records
        )
        grouped: Dict[tuple[str, str], List[Mapping[str, object]]] = {}
        for record in call_records:
            key = (
                str(record.get("step_name", "unknown")),
                str(record.get("prompt_mode", "unknown")),
            )
            grouped.setdefault(key, []).append(record)

        step_rows = []
        for (step_name, prompt_mode), rows in sorted(grouped.items()):
            step_duration_ms = sum(float(row.get("duration_ms", 0.0)) for row in rows)
            step_backend_roundtrip_ms = sum(
                float(row.get("backend_roundtrip_ms", 0.0)) for row in rows
            )
            step_frontend_message_build_ms = sum(
                float(row.get("frontend_message_build_ms", 0.0)) for row in rows
            )
            step_frontend_token_estimate_ms = sum(
                float(row.get("frontend_token_estimate_ms", 0.0)) for row in rows
            )
            step_frontend_overhead_ms = sum(
                float(row.get("frontend_overhead_ms", 0.0)) for row in rows
            )
            step_prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
            step_completion_tokens = sum(
                int(row.get("completion_tokens") or 0) for row in rows
            )
            step_frontend_cache_hits = sum(
                1 for row in rows if bool(row.get("frontend_cache_hit", False))
            )
            step_system_prompt_tokens_estimate = sum(
                int(row.get("system_prompt_tokens_estimate") or 0) for row in rows
            )
            step_user_prompt_tokens_estimate = sum(
                int(row.get("user_prompt_tokens_estimate") or 0) for row in rows
            )
            step_ordered_segment_tokens = sum(
                int(row.get("ordered_segment_tokens") or 0) for row in rows
            )
            step_request_segment_tokens = sum(
                int(row.get("request_segment_tokens") or 0) for row in rows
            )
            step_assembled_prompt_tokens_estimate = sum(
                int(row.get("assembled_prompt_tokens_estimate") or 0) for row in rows
            )
            step_prompt_payload_tokens_estimate = sum(
                int(row.get("prompt_payload_tokens_estimate") or 0) for row in rows
            )
            step_duplicate_prompt_tokens_estimate = sum(
                int(row.get("duplicate_prompt_tokens_estimate") or 0) for row in rows
            )
            step_rows.append(
                {
                    "step_name": step_name,
                    "prompt_mode": prompt_mode,
                    "request_count": len(rows),
                    "total_duration_ms": step_duration_ms,
                    "avg_duration_ms": step_duration_ms / len(rows),
                    "total_backend_roundtrip_ms": step_backend_roundtrip_ms,
                    "avg_backend_roundtrip_ms": step_backend_roundtrip_ms / len(rows),
                    "total_frontend_message_build_ms": step_frontend_message_build_ms,
                    "avg_frontend_message_build_ms": step_frontend_message_build_ms / len(rows),
                    "total_frontend_token_estimate_ms": step_frontend_token_estimate_ms,
                    "avg_frontend_token_estimate_ms": (
                        step_frontend_token_estimate_ms / len(rows)
                    ),
                    "total_frontend_overhead_ms": step_frontend_overhead_ms,
                    "avg_frontend_overhead_ms": step_frontend_overhead_ms / len(rows),
                    "total_prompt_tokens": step_prompt_tokens,
                    "total_completion_tokens": step_completion_tokens,
                    "avg_prompt_tokens": (
                        step_prompt_tokens / len(rows) if rows else 0.0
                    ),
                    "duration_ms_per_1k_prompt_tokens": (
                        step_duration_ms / (step_prompt_tokens / 1000.0)
                        if step_prompt_tokens
                        else 0.0
                    ),
                    "frontend_cache_hits": step_frontend_cache_hits,
                    "frontend_cache_hit_rate": (
                        step_frontend_cache_hits / len(rows) if rows else 0.0
                    ),
                    "system_prompt_tokens_estimate": step_system_prompt_tokens_estimate,
                    "user_prompt_tokens_estimate": step_user_prompt_tokens_estimate,
                    "ordered_segment_tokens": step_ordered_segment_tokens,
                    "request_segment_tokens": step_request_segment_tokens,
                    "assembled_prompt_tokens_estimate": step_assembled_prompt_tokens_estimate,
                    "prompt_payload_tokens_estimate": step_prompt_payload_tokens_estimate,
                    "duplicate_prompt_tokens_estimate": step_duplicate_prompt_tokens_estimate,
                }
            )

        return {
            "request_count": len(call_records),
            "total_duration_ms": total_duration_ms,
            "avg_duration_ms": total_duration_ms / len(call_records),
            "total_backend_roundtrip_ms": total_backend_roundtrip_ms,
            "avg_backend_roundtrip_ms": total_backend_roundtrip_ms / len(call_records),
            "total_frontend_message_build_ms": total_frontend_message_build_ms,
            "avg_frontend_message_build_ms": (
                total_frontend_message_build_ms / len(call_records)
            ),
            "total_frontend_token_estimate_ms": total_frontend_token_estimate_ms,
            "avg_frontend_token_estimate_ms": (
                total_frontend_token_estimate_ms / len(call_records)
            ),
            "total_frontend_overhead_ms": total_frontend_overhead_ms,
            "avg_frontend_overhead_ms": total_frontend_overhead_ms / len(call_records),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_prompt_tokens": total_prompt_tokens / len(call_records),
            "duration_ms_per_1k_prompt_tokens": (
                total_duration_ms / (total_prompt_tokens / 1000.0)
                if total_prompt_tokens
                else 0.0
            ),
            "frontend_cache_hits": frontend_cache_hits,
            "frontend_cache_hit_rate": frontend_cache_hits / len(call_records),
            "total_system_prompt_tokens_estimate": total_system_prompt_tokens_estimate,
            "total_user_prompt_tokens_estimate": total_user_prompt_tokens_estimate,
            "total_ordered_segment_tokens": total_ordered_segment_tokens,
            "total_request_segment_tokens": total_request_segment_tokens,
            "total_assembled_prompt_tokens_estimate": total_assembled_prompt_tokens_estimate,
            "total_prompt_payload_tokens_estimate": total_prompt_payload_tokens_estimate,
            "total_duplicate_prompt_tokens_estimate": total_duplicate_prompt_tokens_estimate,
            "step_rows": step_rows,
        }

    def _write_backend_call_records(
        self,
        *,
        backend_call_path: Path,
        workflow_id: str,
        call_records: Sequence[Mapping[str, object]],
    ) -> None:
        backend_call_path.parent.mkdir(parents=True, exist_ok=True)
        with backend_call_path.open("w", encoding="utf-8") as handle:
            for record in call_records:
                payload = {"workflow_id": workflow_id, **dict(record)}
                handle.write(json.dumps(payload) + "\n")

    def run_instance(self, instance: WorkflowInstance) -> Dict[str, object]:
        trace_path = self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}.jsonl"
        selected_files = self._select_files(instance)
        final_plan_text = ""
        final_patch_text = ""
        final_verdict_text = ""
        status = "UNRESOLVED"

        with TraceLogger(
            trace_path,
            workflow_id=instance.instance_id,
            tenant_id=self.tenant_id,
        ) as trace:
            system_prompt = self._system_prompt_text()
            system_state = trace.create_state(
                state_id="system_v1",
                logical_key="prompt/system",
                state_type="agent_anchor",
                size_bytes=size_bytes(system_prompt),
                token_count=approx_token_count(system_prompt),
                producer="runner",
                recompute_cost=1.0,
                reload_cost=0.5,
                materialization="HBM",
                metadata=module_metadata(
                    context_module="system",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="static",
                    instance_id=instance.instance_id,
                ),
            )
            task_state = trace.create_state(
                state_id="task_v1",
                logical_key="prompt/task",
                state_type="conversation_history",
                size_bytes=size_bytes(instance.problem_statement),
                token_count=approx_token_count(instance.problem_statement),
                producer="runner",
                recompute_cost=1.0,
                reload_cost=0.5,
                materialization="HBM",
                metadata=module_metadata(
                    context_module="task",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="task_fixed",
                ),
            )
            readme_states = self._create_text_states(
                trace=trace,
                state_type="retrieved_document",
                producer="retriever",
                logical_prefix="readme",
                files=instance.readmes,
                materialization="CPU",
                context_module="evidence",
                lifecycle_class="medium",
                is_immutable=True,
                is_shared=True,
                is_ephemeral=False,
                update_cause="retrieval_refresh",
            )
            doc_states = self._create_text_states(
                trace=trace,
                state_type="retrieved_document",
                producer="retriever",
                logical_prefix="retrieval",
                files=selected_files,
                materialization="CPU",
                context_module="evidence",
                lifecycle_class="short",
                is_immutable=True,
                is_shared=False,
                is_ephemeral=False,
                update_cause="retrieval_refresh",
            )

            current_plan_id = None
            current_patch_id = None
            current_verification_id = None
            diagnostic_state_id = None
            live_state_ids = {system_state.state_id, task_state.state_id}
            live_state_ids.update(state.state_id for state in readme_states)
            live_state_ids.update(state.state_id for state in doc_states)

            for iteration in range(1, self.max_iterations + 1):
                planner_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                ]
                planner_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in readme_states
                )
                planner_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in doc_states
                )
                if diagnostic_state_id is not None:
                    planner_segments.append(
                        self._segment(
                            diagnostic_state_id,
                            "scratchpad",
                            role="scratchpad",
                            update_cause="previous_failure",
                        )
                    )
                trace.log_prompt_segments(
                    consumer="planner",
                    prompt_id=f"planner-{iteration}",
                    segments=planner_segments,
                    metadata={"hook": "planner.messages_for_llm", "iteration": iteration},
                )
                plan_text = self.backend.complete(
                    step_name="planner",
                    system_prompt=self._planner_system_prompt(),
                    user_prompt=self._planner_user_prompt(instance, selected_files, iteration, diagnostic_state_id),
                    iteration=iteration,
                    instance=instance,
                )
                final_plan_text = plan_text
                plan_parents = [diagnostic_state_id] if diagnostic_state_id is not None else []
                plan_state = trace.create_state(
                    state_id=f"plan_v{iteration}",
                    logical_key="planner/plan",
                    state_type="plan",
                    size_bytes=size_bytes(plan_text),
                    token_count=approx_token_count(plan_text),
                    producer="planner",
                    parent_state_ids=plan_parents or None,
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="plan",
                        lifecycle_class="medium",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=False,
                        update_cause="replan",
                        iteration=iteration,
                    ),
                )
                live_state_ids.add(plan_state.state_id)
                if current_plan_id is not None:
                    trace.supersede_state(current_plan_id, plan_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_plan_id, consumer="planner", metadata={"reason": "replan"})
                    live_state_ids.discard(current_plan_id)
                current_plan_id = plan_state.state_id

                coder_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_plan_id, "plan", role="plan", update_cause="plan_update"),
                ]
                coder_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in readme_states
                )
                coder_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in doc_states
                )
                trace.log_prompt_segments(
                    consumer="coder",
                    prompt_id=f"coder-{iteration}",
                    segments=coder_segments,
                    metadata={"hook": "messages_for_llm", "iteration": iteration},
                )
                patch_text = self.backend.complete(
                    step_name="coder",
                    system_prompt=self._coder_system_prompt(),
                    user_prompt=self._coder_user_prompt(instance, selected_files, plan_text, iteration),
                    iteration=iteration,
                    instance=instance,
                )
                final_patch_text = patch_text
                patch_parent_ids = [current_plan_id]
                patch_parent_ids.extend(state.state_id for state in doc_states)
                patch_state = trace.create_state(
                    state_id=f"patch_v{iteration}",
                    logical_key="coder/patch",
                    state_type="generated_artifact",
                    size_bytes=size_bytes(patch_text),
                    token_count=approx_token_count(patch_text),
                    producer="coder",
                    parent_state_ids=patch_parent_ids,
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="artifact",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="repatch",
                        iteration=iteration,
                    ),
                )
                live_state_ids.add(patch_state.state_id)
                if current_patch_id is not None:
                    trace.supersede_state(current_patch_id, patch_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_patch_id, consumer="coder", metadata={"reason": "repatch"})
                    live_state_ids.discard(current_patch_id)
                current_patch_id = patch_state.state_id

                tester_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_plan_id, "plan", role="plan", update_cause="plan_update"),
                    self._segment(current_patch_id, "artifact", role="artifact", update_cause="patch_update"),
                ]
                trace.log_prompt_segments(
                    consumer="tester",
                    prompt_id=f"tester-{iteration}",
                    segments=tester_segments,
                    metadata={"hook": "tester.messages_for_llm", "iteration": iteration},
                )
                verdict_text = self.backend.complete(
                    step_name="tester",
                    system_prompt=self._tester_system_prompt(),
                    user_prompt=self._tester_user_prompt(instance, plan_text, patch_text, iteration),
                    iteration=iteration,
                    instance=instance,
                )
                final_verdict_text = verdict_text
                verification_state = trace.create_state(
                    state_id=f"verification_v{iteration}",
                    logical_key="tester/result",
                    state_type="verification_result",
                    size_bytes=size_bytes(verdict_text),
                    token_count=approx_token_count(verdict_text),
                    producer="tester",
                    parent_state_ids=[current_patch_id],
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="verification",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="retest",
                        iteration=iteration,
                    ),
                )
                live_state_ids.add(verification_state.state_id)
                if current_verification_id is not None:
                    trace.supersede_state(
                        current_verification_id,
                        verification_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        current_verification_id,
                        consumer="tester",
                        metadata={"reason": "retest"},
                    )
                    live_state_ids.discard(current_verification_id)
                current_verification_id = verification_state.state_id

                status = self._classify_verdict(verdict_text)
                if status == "RESOLVED":
                    break

                diagnostic_text = self._diagnostic_from_verdict(verdict_text)
                diagnostic_state = trace.create_state(
                    state_id=f"diagnostic_v{iteration}",
                    logical_key="tester/diagnostic",
                    state_type="error_diagnostic",
                    size_bytes=size_bytes(diagnostic_text),
                    token_count=approx_token_count(diagnostic_text),
                    producer="tester",
                    parent_state_ids=[verification_state.state_id],
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="scratchpad",
                        lifecycle_class="very_short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="failure_feedback",
                        iteration=iteration,
                    ),
                )
                live_state_ids.add(diagnostic_state.state_id)
                if diagnostic_state_id is not None:
                    trace.supersede_state(
                        diagnostic_state_id,
                        diagnostic_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        diagnostic_state_id,
                        consumer="planner",
                        metadata={"reason": "new_diagnostic"},
                    )
                    live_state_ids.discard(diagnostic_state_id)
                diagnostic_state_id = diagnostic_state.state_id

            for state_id in sorted(live_state_ids):
                trace.release_state(state_id, consumer="workflow", metadata={"reason": "workflow_end"})

        events = load_trace_events(trace_path)
        validation = validate_trace_events(events)
        return {
            "instance_id": instance.instance_id,
            "status": status,
            "provider": getattr(self.backend, "name", self.backend.__class__.__name__),
            "plan": final_plan_text,
            "patch": final_patch_text,
            "verification": final_verdict_text,
            "trace_path": str(trace_path),
            "trace_validation": {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "summary": validation.summary,
            },
        }

    def _create_text_states(
        self,
        *,
        trace: TraceLogger,
        state_type: str,
        producer: str,
        logical_prefix: str,
        files: Mapping[str, str],
        materialization: str,
        context_module: str,
        lifecycle_class: str,
        is_immutable: bool,
        is_shared: bool,
        is_ephemeral: bool,
        update_cause: str,
    ):
        states = []
        for index, (filename, contents) in enumerate(sorted(files.items()), start=1):
            suffix = sanitize_state_suffix(filename)
            states.append(
                trace.create_state(
                    state_id=f"{logical_prefix}_{index}_{suffix}",
                    logical_key=f"{logical_prefix}/{filename}",
                    state_type=state_type,
                    size_bytes=size_bytes(contents),
                    token_count=approx_token_count(contents),
                    producer=producer,
                    materialization=materialization,
                    metadata=module_metadata(
                        context_module=context_module,
                        lifecycle_class=lifecycle_class,
                        is_immutable=is_immutable,
                        is_shared=is_shared,
                        is_ephemeral=is_ephemeral,
                        update_cause=update_cause,
                        path=filename,
                    ),
                )
            )
        return states

    def _select_files(self, instance: WorkflowInstance) -> Dict[str, str]:
        items = list(sorted(instance.file_contents.items()))[: self.max_files]
        return dict(items)

    def _segment(
        self,
        state_id: str,
        context_module: str,
        *,
        role: str,
        update_cause: str | None = None,
    ) -> Dict[str, object]:
        return {
            "state_id": state_id,
            "context_module": context_module,
            "segment_role": role,
            "update_cause": update_cause or "read",
        }

    def _system_prompt_text(self) -> str:
        return (
            "You are a software engineering agent. "
            "Read the task, reason carefully, and propose the smallest correct change."
        )

    def _planner_system_prompt(self) -> str:
        return (
            "You are a software planning agent. Produce a short, actionable repair plan "
            "that references the likely bug, target files, and validation goal."
        )

    def _planner_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diag_text = (
            f"Previous diagnostic state: {diagnostic_state_id}\n"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            f"{diag_text}"
            f"Available files:\n"
            + "\n".join(f"- {path}" for path in sorted(selected_files))
        )

    def _coder_system_prompt(self) -> str:
        return (
            "You are a code-writing agent. Return a single unified diff patch that is directly applicable with git apply."
        )

    def _coder_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        plan_text: str,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Repository context:\n{make_code_text(selected_files)}\n"
        )

    def _coder_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            "Use the current routed context, active plan, and repository evidence to produce a minimal unified diff patch.\n"
        )

    def _tester_system_prompt(self) -> str:
        return (
            "You are a verification agent. Decide if the patch resolves the issue. "
            "Respond with PASS or FAIL on the first line, then justify briefly."
        )

    def _tester_user_prompt(
        self,
        instance: WorkflowInstance,
        plan_text: str,
        patch_text: str,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Patch:\n{patch_text}\n"
        )

    def _tester_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            "Evaluate the current patch against the active plan and return PASS or FAIL.\n"
        )

    def _classify_verdict(self, verdict_text: str) -> str:
        first_line = verdict_text.splitlines()[0].upper() if verdict_text else ""
        if any(token in first_line for token in PASS_TOKENS):
            return "RESOLVED"
        if any(token in first_line for token in FAIL_TOKENS):
            return "UNRESOLVED"
        return "UNRESOLVED"

    def _diagnostic_from_verdict(self, verdict_text: str) -> str:
        lines = verdict_text.splitlines()
        if len(lines) <= 1:
            return truncate_text(verdict_text, 400)
        return truncate_text("\n".join(lines[1:]), 400)

    def _stress_planner_system_prompt(self) -> str:
        return (
            "You are a long-horizon planning agent. Refine the current repair strategy "
            "using persistent evidence and the latest critique."
        )

    def _stress_planner_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Stress iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            "Revise the current plan while preserving stable context."
        )

    def _stress_critic_system_prompt(self) -> str:
        return (
            "You are a critic agent. Review the current plan and identify one concrete "
            "risk or improvement before the next iteration."
        )

    def _stress_critic_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Stress iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            "Provide concise critique to drive the next plan revision."
        )

    def _research_planner_system_prompt(self) -> str:
        return (
            "You are a deep research planner. Maintain a stable survey outline while "
            "integrating persistent evidence and the latest critique."
        )

    def _research_planner_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Research iteration: {iteration}\n"
            f"Question:\n{instance.problem_statement}\n\n"
            "Revise the outline and identify the most important evidence to emphasize next."
        )

    def _research_reader_system_prompt(self) -> str:
        return (
            "You are a research reader. Extract concise systems-relevant findings from the "
            "currently focused evidence."
        )

    def _research_reader_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
        source_name: str,
    ) -> str:
        return (
            f"Research iteration: {iteration}\n"
            f"Question:\n{instance.problem_statement}\n\n"
            f"Focused source: {source_name}\n"
            "Produce a short note that can be reused by later planner and writer steps."
        )

    def _research_writer_system_prompt(self) -> str:
        return (
            "You are a technical writer. Expand the current survey draft using the active "
            "plan and accumulated reading notes."
        )

    def _research_writer_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Research iteration: {iteration}\n"
            f"Question:\n{instance.problem_statement}\n\n"
            "Write the next concise survey draft while preserving the stable outline."
        )

    def _research_critic_system_prompt(self) -> str:
        return (
            "You are a critic. Review the current survey draft and identify the next high-value "
            "revision needed for clarity or systems rigor."
        )

    def _research_critic_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Research iteration: {iteration}\n"
            f"Question:\n{instance.problem_statement}\n\n"
            "Provide one concise critique that should influence the next planning pass."
        )


class LangGraphStyleTracedAgentRunner(TracedAgentRunner):
    def run_instance(self, instance: WorkflowInstance) -> Dict[str, object]:
        trace_path = self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}.jsonl"
        selected_files = self._select_files(instance)
        final_plan_text = ""
        final_patch_text = ""
        final_verdict_text = ""
        status = "UNRESOLVED"

        with TraceLogger(
            trace_path,
            workflow_id=instance.instance_id,
            tenant_id=self.tenant_id,
        ) as trace:
            system_prompt = self._system_prompt_text()
            system_state = trace.create_state(
                state_id="system_v1",
                logical_key="prompt/system",
                state_type="agent_anchor",
                size_bytes=size_bytes(system_prompt),
                token_count=approx_token_count(system_prompt),
                producer="runner",
                recompute_cost=1.0,
                reload_cost=0.5,
                materialization="HBM",
                metadata=module_metadata(
                    context_module="system",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="static",
                    agent_family="langgraph_style",
                ),
            )
            task_state = trace.create_state(
                state_id="task_v1",
                logical_key="prompt/task",
                state_type="conversation_history",
                size_bytes=size_bytes(instance.problem_statement),
                token_count=approx_token_count(instance.problem_statement),
                producer="runner",
                recompute_cost=1.0,
                reload_cost=0.5,
                materialization="HBM",
                metadata=module_metadata(
                    context_module="task",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="task_fixed",
                    agent_family="langgraph_style",
                ),
            )
            readme_states = self._create_text_states(
                trace=trace,
                state_type="retrieved_document",
                producer="retriever",
                logical_prefix="readme",
                files=instance.readmes,
                materialization="CPU",
                context_module="evidence",
                lifecycle_class="medium",
                is_immutable=True,
                is_shared=True,
                is_ephemeral=False,
                update_cause="retrieval_refresh",
            )
            doc_states = self._create_text_states(
                trace=trace,
                state_type="retrieved_document",
                producer="retriever",
                logical_prefix="retrieval",
                files=selected_files,
                materialization="CPU",
                context_module="evidence",
                lifecycle_class="short",
                is_immutable=True,
                is_shared=False,
                is_ephemeral=False,
                update_cause="retrieval_refresh",
            )

            current_route_id = None
            current_plan_id = None
            current_patch_id = None
            current_review_id = None
            current_verification_id = None
            diagnostic_state_id = None
            live_state_ids = {system_state.state_id, task_state.state_id}
            live_state_ids.update(state.state_id for state in readme_states)
            live_state_ids.update(state.state_id for state in doc_states)

            for iteration in range(1, self.max_iterations + 1):
                router_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                ]
                router_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in readme_states
                )
                router_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in doc_states
                )
                if diagnostic_state_id is not None:
                    router_segments.append(
                        self._segment(
                            diagnostic_state_id,
                            "scratchpad",
                            role="scratchpad",
                            update_cause="failure_feedback",
                        )
                    )
                trace.log_prompt_segments(
                    consumer="router",
                    prompt_id=f"router-{iteration}",
                    segments=router_segments,
                    metadata={"hook": "router.messages_for_llm", "iteration": iteration},
                )
                route_text = self.backend.complete(
                    step_name="router",
                    system_prompt=self._router_system_prompt(),
                    user_prompt=self._router_user_prompt(instance, selected_files, iteration, diagnostic_state_id),
                    iteration=iteration,
                    instance=instance,
                )
                route_state = trace.create_state(
                    state_id=f"route_v{iteration}",
                    logical_key="router/decision",
                    state_type="summary",
                    size_bytes=size_bytes(route_text),
                    token_count=approx_token_count(route_text),
                    producer="router",
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="summary",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="reroute",
                        iteration=iteration,
                        agent_family="langgraph_style",
                    ),
                )
                live_state_ids.add(route_state.state_id)
                if current_route_id is not None:
                    trace.supersede_state(current_route_id, route_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_route_id, consumer="router", metadata={"reason": "reroute"})
                    live_state_ids.discard(current_route_id)
                current_route_id = route_state.state_id

                planner_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_route_id, "summary", role="router"),
                ]
                planner_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in readme_states
                )
                planner_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in doc_states
                )
                if diagnostic_state_id is not None:
                    planner_segments.append(
                        self._segment(
                            diagnostic_state_id,
                            "scratchpad",
                            role="scratchpad",
                            update_cause="previous_failure",
                        )
                    )
                trace.log_prompt_segments(
                    consumer="planner",
                    prompt_id=f"planner-{iteration}",
                    segments=planner_segments,
                    metadata={"hook": "planner.messages_for_llm", "iteration": iteration},
                )
                plan_text = self.backend.complete(
                    step_name="planner",
                    system_prompt=self._planner_system_prompt(),
                    user_prompt=self._planner_user_prompt(instance, selected_files, iteration, diagnostic_state_id),
                    iteration=iteration,
                    instance=instance,
                )
                final_plan_text = plan_text
                plan_parent_ids = [current_route_id]
                if diagnostic_state_id is not None:
                    plan_parent_ids.append(diagnostic_state_id)
                plan_state = trace.create_state(
                    state_id=f"plan_v{iteration}",
                    logical_key="planner/plan",
                    state_type="plan",
                    size_bytes=size_bytes(plan_text),
                    token_count=approx_token_count(plan_text),
                    producer="planner",
                    parent_state_ids=plan_parent_ids,
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="plan",
                        lifecycle_class="medium",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=False,
                        update_cause="replan",
                        iteration=iteration,
                        agent_family="langgraph_style",
                    ),
                )
                live_state_ids.add(plan_state.state_id)
                if current_plan_id is not None:
                    trace.supersede_state(current_plan_id, plan_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_plan_id, consumer="planner", metadata={"reason": "replan"})
                    live_state_ids.discard(current_plan_id)
                current_plan_id = plan_state.state_id

                coder_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_route_id, "summary", role="router"),
                    self._segment(current_plan_id, "plan", role="plan", update_cause="plan_update"),
                ]
                coder_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in readme_states
                )
                coder_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in doc_states
                )
                trace.log_prompt_segments(
                    consumer="coder",
                    prompt_id=f"coder-{iteration}",
                    segments=coder_segments,
                    metadata={"hook": "coder.messages_for_llm", "iteration": iteration},
                )
                patch_text = self.backend.complete(
                    step_name="coder",
                    system_prompt=self._coder_system_prompt(),
                    user_prompt=self._coder_user_prompt(instance, selected_files, plan_text, iteration),
                    iteration=iteration,
                    instance=instance,
                )
                final_patch_text = patch_text
                patch_parent_ids = [current_route_id, current_plan_id]
                patch_parent_ids.extend(state.state_id for state in doc_states)
                patch_state = trace.create_state(
                    state_id=f"patch_v{iteration}",
                    logical_key="coder/patch",
                    state_type="generated_artifact",
                    size_bytes=size_bytes(patch_text),
                    token_count=approx_token_count(patch_text),
                    producer="coder",
                    parent_state_ids=patch_parent_ids,
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="artifact",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="repatch",
                        iteration=iteration,
                        agent_family="langgraph_style",
                    ),
                )
                live_state_ids.add(patch_state.state_id)
                if current_patch_id is not None:
                    trace.supersede_state(current_patch_id, patch_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_patch_id, consumer="coder", metadata={"reason": "repatch"})
                    live_state_ids.discard(current_patch_id)
                current_patch_id = patch_state.state_id

                reviewer_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_route_id, "summary", role="router"),
                    self._segment(current_plan_id, "plan", role="plan", update_cause="plan_update"),
                    self._segment(current_patch_id, "artifact", role="artifact", update_cause="patch_update"),
                ]
                trace.log_prompt_segments(
                    consumer="reviewer",
                    prompt_id=f"reviewer-{iteration}",
                    segments=reviewer_segments,
                    metadata={"hook": "reviewer.messages_for_llm", "iteration": iteration},
                )
                review_text = self.backend.complete(
                    step_name="reviewer",
                    system_prompt=self._reviewer_system_prompt(),
                    user_prompt=self._reviewer_user_prompt(instance, plan_text, patch_text, iteration),
                    iteration=iteration,
                    instance=instance,
                )
                review_state = trace.create_state(
                    state_id=f"review_v{iteration}",
                    logical_key="reviewer/notes",
                    state_type="summary",
                    size_bytes=size_bytes(review_text),
                    token_count=approx_token_count(review_text),
                    producer="reviewer",
                    parent_state_ids=[current_patch_id],
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="summary",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="review_update",
                        iteration=iteration,
                        agent_family="langgraph_style",
                    ),
                )
                live_state_ids.add(review_state.state_id)
                if current_review_id is not None:
                    trace.supersede_state(current_review_id, review_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_review_id, consumer="reviewer", metadata={"reason": "rereview"})
                    live_state_ids.discard(current_review_id)
                current_review_id = review_state.state_id

                tester_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_patch_id, "artifact", role="artifact", update_cause="patch_update"),
                    self._segment(current_review_id, "summary", role="review"),
                ]
                trace.log_prompt_segments(
                    consumer="tester",
                    prompt_id=f"tester-{iteration}",
                    segments=tester_segments,
                    metadata={"hook": "tester.messages_for_llm", "iteration": iteration},
                )
                verdict_text = self.backend.complete(
                    step_name="tester",
                    system_prompt=self._tester_system_prompt(),
                    user_prompt=self._tester_user_prompt(instance, plan_text, patch_text, iteration),
                    iteration=iteration,
                    instance=instance,
                )
                final_verdict_text = verdict_text
                verification_state = trace.create_state(
                    state_id=f"verification_v{iteration}",
                    logical_key="tester/result",
                    state_type="verification_result",
                    size_bytes=size_bytes(verdict_text),
                    token_count=approx_token_count(verdict_text),
                    producer="tester",
                    parent_state_ids=[current_patch_id, current_review_id],
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="verification",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="retest",
                        iteration=iteration,
                        agent_family="langgraph_style",
                    ),
                )
                live_state_ids.add(verification_state.state_id)
                if current_verification_id is not None:
                    trace.supersede_state(
                        current_verification_id,
                        verification_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        current_verification_id,
                        consumer="tester",
                        metadata={"reason": "retest"},
                    )
                    live_state_ids.discard(current_verification_id)
                current_verification_id = verification_state.state_id

                status = self._classify_verdict(verdict_text)
                if status == "RESOLVED":
                    break

                diagnostic_text = self._diagnostic_from_verdict(verdict_text)
                diagnostic_state = trace.create_state(
                    state_id=f"diagnostic_v{iteration}",
                    logical_key="tester/diagnostic",
                    state_type="error_diagnostic",
                    size_bytes=size_bytes(diagnostic_text),
                    token_count=approx_token_count(diagnostic_text),
                    producer="tester",
                    parent_state_ids=[verification_state.state_id],
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="scratchpad",
                        lifecycle_class="very_short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="failure_feedback",
                        iteration=iteration,
                        agent_family="langgraph_style",
                    ),
                )
                live_state_ids.add(diagnostic_state.state_id)
                if diagnostic_state_id is not None:
                    trace.supersede_state(
                        diagnostic_state_id,
                        diagnostic_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        diagnostic_state_id,
                        consumer="router",
                        metadata={"reason": "new_diagnostic"},
                    )
                    live_state_ids.discard(diagnostic_state_id)
                diagnostic_state_id = diagnostic_state.state_id

            for state_id in sorted(live_state_ids):
                trace.release_state(state_id, consumer="workflow", metadata={"reason": "workflow_end"})

        events = load_trace_events(trace_path)
        validation = validate_trace_events(events)
        return {
            "instance_id": instance.instance_id,
            "status": status,
            "provider": getattr(self.backend, "name", self.backend.__class__.__name__),
            "agent_family": "langgraph_style",
            "plan": final_plan_text,
            "patch": final_patch_text,
            "verification": final_verdict_text,
            "trace_path": str(trace_path),
            "trace_validation": {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "summary": validation.summary,
            },
        }

    def _router_system_prompt(self) -> str:
        return (
            "You are a routing agent. Decide whether the issue needs plan-then-patch "
            "or direct patching, and briefly justify the route."
        )

    def _router_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            f"Previous failure context: {diagnostic_state_id}\n"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            f"{diagnostic_text}"
            f"Candidate files:\n"
            + "\n".join(f"- {path}" for path in sorted(selected_files))
        )

    def _router_instruction_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            "Previous failure context is available as runtime scratch context.\n"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            f"{diagnostic_text}"
            f"Candidate files:\n"
            + "\n".join(f"- {path}" for path in sorted(selected_files))
        )

    def _router_instruction_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            "Previous failure context is available as runtime scratch context.\n"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            f"{diagnostic_text}"
            f"Candidate files:\n"
            + "\n".join(f"- {path}" for path in sorted(selected_files))
        )

    def _router_instruction_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            f"Previous failure context segment: {diagnostic_state_id}\n"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"Iteration: {iteration}\n"
            "Route the issue using the already provided context segments.\n"
            f"{diagnostic_text}"
            "Candidate files:\n"
            + "\n".join(f"- {path}" for path in sorted(selected_files))
        )

    def _reviewer_system_prompt(self) -> str:
        return (
            "You are a review agent. Inspect the proposed patch, identify residual risks, "
            "and summarize whether it is ready for verification."
        )

    def _reviewer_user_prompt(
        self,
        instance: WorkflowInstance,
        plan_text: str,
        patch_text: str,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Patch under review:\n{patch_text}\n"
        )

    def _reviewer_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            "Review the candidate patch using the provided context segments and decide "
            "whether it is ready for verification."
        )
class _OpenSWEAgentPromptBackedMixin:
    def __init__(
        self,
        *,
        backend: ModelBackend,
        trace_dir: str | Path,
        tenant_id: str = "local",
        max_iterations: int = 2,
        max_files: int = 5,
        prompt_runtime_mode: str = "monolithic",
        swe_agent_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            backend=backend,
            trace_dir=trace_dir,
            tenant_id=tenant_id,
            max_iterations=max_iterations,
            max_files=max_files,
            prompt_runtime_mode=prompt_runtime_mode,
        )
        self._swe_agent_prompts = load_swe_agent_prompt_pack(swe_agent_path)

    def _agent_family_name(self) -> str:
        return "swe_agent"

    def _result_workload_metadata(self) -> Dict[str, object]:
        return {
            "workload_source": "swe_agent",
            "swe_agent_path": str(self._swe_agent_prompts.repo_path),
        }

    def _swe_agent_working_dir(self, instance: WorkflowInstance) -> str:
        return f"/workspace/{sanitize_state_suffix(instance.instance_id)}"

    def _swe_agent_instance_context(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
    ) -> str:
        rendered = _render_template_variables(
            self._swe_agent_prompts.instance_template,
            working_dir=self._swe_agent_working_dir(instance),
            problem_statement=instance.problem_statement,
        )
        if not selected_files:
            return rendered
        return (
            f"{rendered}\n\n"
            "Relevant files currently loaded into context:\n"
            + "\n".join(f"- {path}" for path in sorted(selected_files))
        )

    def _swe_agent_observation(self, observation: str) -> str:
        template = (
            self._swe_agent_prompts.next_step_template
            if observation.strip()
            else self._swe_agent_prompts.next_step_no_output_template
        )
        return _render_template_variables(template, observation=observation)

    def _swe_agent_review_message(self, patch_text: str) -> str:
        template = (
            self._swe_agent_prompts.submit_review_messages[0]
            if self._swe_agent_prompts.submit_review_messages
            else (
                "Review the current patch carefully, verify that it addresses the issue, "
                "and identify any residual risk before submission.\n\n<diff>\n{{diff}}\n</diff>"
            )
        )
        return _render_template_variables(template, diff=patch_text)

    def _system_prompt_text(self) -> str:
        return self._swe_agent_prompts.system_template

    def _router_system_prompt(self) -> str:
        return self._swe_agent_prompts.system_template

    def _router_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            f"\n\nPrevious failure context:\n{diagnostic_state_id}"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"{self._swe_agent_instance_context(instance, selected_files)}{diagnostic_text}\n\n"
            f"Iteration: {iteration}\n"
            "Current phase: inspect the issue and decide the next repair route."
        )

    def _router_instruction_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        return (
            f"{self._swe_agent_instance_context(instance, selected_files)}\n\n"
            f"{self._swe_agent_observation('Use the already provided context segments to choose the next repair route.')}\n"
            f"Iteration: {iteration}\n"
            + (
                f"Previous failure context segment: {diagnostic_state_id}\n"
                if diagnostic_state_id is not None
                else ""
            )
        )

    def _planner_system_prompt(self) -> str:
        return self._swe_agent_prompts.system_template

    def _planner_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            f"\n\nPrevious failure context:\n{diagnostic_state_id}"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"{self._swe_agent_instance_context(instance, selected_files)}{diagnostic_text}\n\n"
            f"Iteration: {iteration}\n"
            "Current phase: produce a concise repair plan before editing."
        )

    def _coder_system_prompt(self) -> str:
        return self._swe_agent_prompts.system_template

    def _coder_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        plan_text: str,
        iteration: int,
    ) -> str:
        return (
            f"{self._swe_agent_instance_context(instance, selected_files)}\n\n"
            f"Iteration: {iteration}\n"
            f"Plan:\n{plan_text}\n\n"
            "Current phase: produce the minimal unified diff patch."
        )

    def _coder_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Issue:\n{instance.problem_statement}\n\n"
            f"{self._swe_agent_observation('Use the active route, plan, and repository context to produce the next patch revision.')}\n"
            f"Iteration: {iteration}\n"
        )

    def _reviewer_system_prompt(self) -> str:
        return self._swe_agent_prompts.system_template

    def _reviewer_user_prompt(
        self,
        instance: WorkflowInstance,
        plan_text: str,
        patch_text: str,
        iteration: int,
    ) -> str:
        return (
            f"Issue:\n{instance.problem_statement}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"{self._swe_agent_review_message(patch_text)}\n\n"
            f"Iteration: {iteration}\n"
        )

    def _reviewer_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Issue:\n{instance.problem_statement}\n\n"
            f"{self._swe_agent_observation('Review the active patch and summarize whether it is ready for verification.')}\n"
            f"Iteration: {iteration}\n"
        )

    def _tester_system_prompt(self) -> str:
        return (
            f"{self._swe_agent_prompts.system_template}\n\n"
            "When verifying a candidate patch, respond with PASS or FAIL on the first line."
        )

    def _tester_user_prompt(
        self,
        instance: WorkflowInstance,
        plan_text: str,
        patch_text: str,
        iteration: int,
    ) -> str:
        return (
            f"Issue:\n{instance.problem_statement}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Patch:\n{patch_text}\n\n"
            f"Iteration: {iteration}\n"
            "Decide whether the patch is ready to submit."
        )

    def _tester_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Issue:\n{instance.problem_statement}\n\n"
            f"{self._swe_agent_observation('Verify the active patch against the issue and return PASS or FAIL.')}\n"
            f"Iteration: {iteration}\n"
        )

    def _reviewer_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            "Review the active patch against the active plan and summarize residual risk.\n"
        )

    def _reviewer_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            "Review the active patch against the active plan and summarize residual risk.\n"
        )


class _OpenHandsPromptBackedMixin:
    def __init__(
        self,
        *,
        backend: ModelBackend,
        trace_dir: str | Path,
        tenant_id: str = "local",
        max_iterations: int = 2,
        max_files: int = 5,
        prompt_runtime_mode: str = "monolithic",
        openhands_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            backend=backend,
            trace_dir=trace_dir,
            tenant_id=tenant_id,
            max_iterations=max_iterations,
            max_files=max_files,
            prompt_runtime_mode=prompt_runtime_mode,
        )
        self._openhands_prompts = load_openhands_prompt_pack(openhands_path)

    def _agent_family_name(self) -> str:
        return "openhands"

    def _result_workload_metadata(self) -> Dict[str, object]:
        return {
            "workload_source": "openhands",
            "openhands_path": str(self._openhands_prompts.repo_path),
            "openhands_default_agent": self._openhands_prompts.default_agent_name,
        }

    def _openhands_working_dir(self, instance: WorkflowInstance) -> str:
        return f"/workspace/{sanitize_state_suffix(instance.instance_id)}"

    def _openhands_guidance_excerpt(self) -> str:
        agents_excerpt = truncate_text(
            self._openhands_prompts.agents_guidance.strip().replace("\r\n", "\n"),
            1400,
        )
        development_excerpt = truncate_text(
            self._openhands_prompts.development_guidance.strip().replace("\r\n", "\n"),
            900,
        )
        excerpt = agents_excerpt
        if development_excerpt:
            excerpt += f"\n\nDevelopment notes:\n{development_excerpt}"
        return excerpt

    def _openhands_instance_context(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
    ) -> str:
        context = (
            f"OpenHands working directory: {self._openhands_working_dir(instance)}\n"
            f"Issue:\n{instance.problem_statement}\n"
        )
        if selected_files:
            context += (
                "\nCandidate files already loaded into context:\n"
                + "\n".join(f"- {path}" for path in sorted(selected_files))
            )
        return context

    def _system_prompt_text(self) -> str:
        return (
            f"You are operating in an OpenHands-style {self._openhands_prompts.default_agent_name} "
            "software-engineering workflow. Follow the repository guidance below while "
            "planning, patching, reviewing, and verifying.\n\n"
            f"{self._openhands_guidance_excerpt()}"
        )

    def _router_system_prompt(self) -> str:
        return self._system_prompt_text()

    def _router_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            f"\n\nPrevious failure context:\n{diagnostic_state_id}"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"{self._openhands_instance_context(instance, selected_files)}{diagnostic_text}\n\n"
            f"Iteration: {iteration}\n"
            "Current phase: choose the next OpenHands repair route."
        )

    def _router_instruction_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        return (
            f"{self._openhands_instance_context(instance, selected_files)}\n\n"
            f"Iteration: {iteration}\n"
            + (
                f"Previous failure context segment: {diagnostic_state_id}\n"
                if diagnostic_state_id is not None
                else ""
            )
            + "Use the provided context segments to choose the next OpenHands route."
        )

    def _planner_system_prompt(self) -> str:
        return self._system_prompt_text()

    def _planner_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            f"\n\nPrevious failure context:\n{diagnostic_state_id}"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"{self._openhands_instance_context(instance, selected_files)}{diagnostic_text}\n\n"
            f"Iteration: {iteration}\n"
            "Current phase: produce the next concise CodeAct-style repair plan."
        )

    def _coder_system_prompt(self) -> str:
        return self._system_prompt_text()

    def _coder_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        plan_text: str,
        iteration: int,
    ) -> str:
        return (
            f"{self._openhands_instance_context(instance, selected_files)}\n\n"
            f"Iteration: {iteration}\n"
            f"Plan:\n{plan_text}\n\n"
            "Current phase: produce the minimal patch needed to advance the fix."
        )

    def _coder_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Issue:\n{instance.problem_statement}\n\n"
            f"Iteration: {iteration}\n"
            "Use the active route, plan, and repository context segments to produce the next "
            "OpenHands patch revision."
        )

    def _reviewer_system_prompt(self) -> str:
        return self._system_prompt_text()

    def _reviewer_user_prompt(
        self,
        instance: WorkflowInstance,
        plan_text: str,
        patch_text: str,
        iteration: int,
    ) -> str:
        return (
            f"{self._openhands_instance_context(instance, {})}\n\n"
            f"Iteration: {iteration}\n"
            f"Plan:\n{plan_text}\n\n"
            f"Patch under review:\n{patch_text}\n"
        )

    def _reviewer_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Issue:\n{instance.problem_statement}\n\n"
            f"Iteration: {iteration}\n"
            "Review the candidate patch using the provided context segments and identify "
            "remaining OpenHands-style execution risks before verification."
        )

    def _tester_system_prompt(self) -> str:
        return (
            f"{self._system_prompt_text()}\n\n"
            "When verifying a candidate patch, respond with PASS or FAIL on the first line."
        )

    def _tester_user_prompt(
        self,
        instance: WorkflowInstance,
        plan_text: str,
        patch_text: str,
        iteration: int,
    ) -> str:
        return (
            f"{self._openhands_instance_context(instance, {})}\n\n"
            f"Iteration: {iteration}\n"
            f"Plan:\n{plan_text}\n\n"
            f"Patch:\n{patch_text}\n\n"
            "Current phase: judge whether the patch resolves the issue and summarize the "
            "verification outcome."
        )

    def _tester_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Issue:\n{instance.problem_statement}\n\n"
            f"Iteration: {iteration}\n"
            "Use the active patch and review context segments to decide whether the issue is "
            "resolved and what failure diagnostics should persist."
        )


class SyntheticStressTracedAgentRunner(TracedAgentRunner):
    def run_instance(self, instance: WorkflowInstance) -> Dict[str, object]:
        trace_path = self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}.jsonl"
        runtime_event_path = (
            self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}_runtime.jsonl"
        )
        backend_call_path = (
            self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}_backend_calls.jsonl"
        )
        selected_files = self._select_files(instance)
        self._reset_monolithic_prompt_runtime_state()

        with TraceLogger(
            trace_path,
            workflow_id=instance.instance_id,
            tenant_id=self.tenant_id,
        ) as trace:
            materializer = SegmentRuntime(
                reclaim_grace_by_role=self._reclaim_grace_by_role,
            )
            system_prompt = self._system_prompt_text()
            system_state = trace.create_state(
                state_id="system_v1",
                logical_key="prompt/system",
                state_type="agent_anchor",
                size_bytes=size_bytes(system_prompt),
                token_count=approx_token_count(system_prompt),
                producer="runner",
                materialization="HBM",
                metadata=module_metadata(
                    context_module="system",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="static",
                    agent_family="synthetic_stress",
                ),
            )
            self._register_runtime_segment(
                materializer=materializer,
                handle=system_state,
                workflow_id=instance.instance_id,
                module="system",
                role="system",
                text=system_prompt,
                materialization="HBM",
                is_shared=True,
                is_immutable=True,
                is_ephemeral=False,
                metadata={"agent_family": "synthetic_stress"},
            )
            task_state = trace.create_state(
                state_id="task_v1",
                logical_key="prompt/task",
                state_type="conversation_history",
                size_bytes=size_bytes(instance.problem_statement),
                token_count=approx_token_count(instance.problem_statement),
                producer="runner",
                materialization="HBM",
                metadata=module_metadata(
                    context_module="task",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="task_fixed",
                    agent_family="synthetic_stress",
                ),
            )
            self._register_runtime_segment(
                materializer=materializer,
                handle=task_state,
                workflow_id=instance.instance_id,
                module="task",
                role="task",
                text=instance.problem_statement,
                materialization="HBM",
                is_shared=True,
                is_immutable=True,
                is_ephemeral=False,
                metadata={"agent_family": "synthetic_stress"},
            )
            evidence_states = self._create_text_states(
                trace=trace,
                state_type="retrieved_document",
                producer="retriever",
                logical_prefix="stress_retrieval",
                files=selected_files,
                materialization="CPU",
                context_module="evidence",
                lifecycle_class="long",
                is_immutable=True,
                is_shared=True,
                is_ephemeral=False,
                update_cause="retrieval_refresh",
            )
            for state, (_, contents) in zip(evidence_states, sorted(selected_files.items())):
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=state,
                    workflow_id=instance.instance_id,
                    module="evidence",
                    role="evidence",
                    text=contents,
                    materialization="CPU",
                    is_shared=True,
                    is_immutable=True,
                    is_ephemeral=False,
                    metadata={"agent_family": "synthetic_stress"},
                )

            live_state_ids = {system_state.state_id, task_state.state_id}
            live_state_ids.update(state.state_id for state in evidence_states)
            current_plan_id: str | None = None
            current_critique_id: str | None = None
            final_plan_text = ""
            final_critique_text = ""

            for iteration in range(1, self.max_iterations + 1):
                planner_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                ]
                planner_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in evidence_states
                )
                if current_critique_id is not None:
                    planner_segments.append(
                        self._segment(
                            current_critique_id,
                            "scratchpad",
                            role="scratchpad",
                            update_cause="stress_feedback",
                        )
                    )
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="stress_planner",
                    step_name="planner",
                    prompt_id=f"stress-planner-{iteration}",
                    system_prompt=self._stress_planner_system_prompt(),
                    user_prompt=self._stress_planner_user_prompt(instance, iteration),
                    segments=planner_segments,
                    metadata={"hook": "stress.planner", "iteration": iteration},
                )
                plan_text = self.backend.complete(
                    step_name="planner",
                    system_prompt=self._stress_planner_system_prompt(),
                    user_prompt=self._stress_planner_user_prompt(instance, iteration),
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                plan_state = trace.create_state(
                    state_id=f"stress_plan_v{iteration}",
                    logical_key="stress/planner/plan",
                    state_type="plan",
                    size_bytes=size_bytes(plan_text),
                    token_count=approx_token_count(plan_text),
                    producer="stress_planner",
                    parent_state_ids=(
                        [current_critique_id] if current_critique_id is not None else None
                    ),
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="plan",
                        lifecycle_class="medium",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=False,
                        update_cause="stress_replan",
                        iteration=iteration,
                        agent_family="synthetic_stress",
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=plan_state,
                    workflow_id=instance.instance_id,
                    module="plan",
                    role="plan",
                    text=plan_text,
                    materialization="HBM",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=False,
                    metadata={"agent_family": "synthetic_stress"},
                )
                live_state_ids.add(plan_state.state_id)
                if current_plan_id is not None:
                    trace.supersede_state(
                        current_plan_id,
                        plan_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        current_plan_id,
                        consumer="stress_planner",
                        metadata={"reason": "stress_replan"},
                    )
                    self._release_runtime_segment(
                        materializer=materializer,
                        state_id=current_plan_id,
                    )
                    live_state_ids.discard(current_plan_id)
                current_plan_id = plan_state.state_id
                final_plan_text = plan_text

                critic_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_plan_id, "plan", role="plan"),
                ]
                critic_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in evidence_states
                )
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="stress_critic",
                    step_name="reviewer",
                    prompt_id=f"stress-critic-{iteration}",
                    system_prompt=self._stress_critic_system_prompt(),
                    user_prompt=self._stress_critic_user_prompt(instance, iteration),
                    segments=critic_segments,
                    metadata={"hook": "stress.critic", "iteration": iteration},
                )
                critique_text = self.backend.complete(
                    step_name="reviewer",
                    system_prompt=self._stress_critic_system_prompt(),
                    user_prompt=self._stress_critic_user_prompt(instance, iteration),
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                critique_state = trace.create_state(
                    state_id=f"stress_critique_v{iteration}",
                    logical_key="stress/critic/notes",
                    state_type="error_diagnostic",
                    size_bytes=size_bytes(critique_text),
                    token_count=approx_token_count(critique_text),
                    producer="stress_critic",
                    parent_state_ids=[current_plan_id],
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="scratchpad",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="stress_critique",
                        iteration=iteration,
                        agent_family="synthetic_stress",
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=critique_state,
                    workflow_id=instance.instance_id,
                    module="scratchpad",
                    role="scratchpad",
                    text=critique_text,
                    materialization="CPU",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=True,
                    metadata={"agent_family": "synthetic_stress"},
                )
                live_state_ids.add(critique_state.state_id)
                if current_critique_id is not None:
                    trace.supersede_state(
                        current_critique_id,
                        critique_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        current_critique_id,
                        consumer="stress_critic",
                        metadata={"reason": "stress_refresh"},
                    )
                    self._release_runtime_segment(
                        materializer=materializer,
                        state_id=current_critique_id,
                    )
                    live_state_ids.discard(current_critique_id)
                current_critique_id = critique_state.state_id
                final_critique_text = critique_text

            for state_id in sorted(live_state_ids):
                trace.release_state(state_id, consumer="workflow", metadata={"reason": "workflow_end"})
                self._release_runtime_segment(materializer=materializer, state_id=state_id)
            if self.prompt_runtime_mode == "monolithic":
                self._release_monolithic_prompt_runtime(materializer=materializer)

        events = load_trace_events(trace_path)
        validation = validate_trace_events(events)
        runtime_summary = self._runtime_event_summary(materializer)
        backend_call_records = self.backend.drain_call_records()
        backend_call_summary = self._backend_call_summary(backend_call_records)
        self._write_runtime_events(
            materializer=materializer,
            runtime_event_path=runtime_event_path,
            workflow_id=instance.instance_id,
            agent_family="synthetic_stress",
        )
        self._write_backend_call_records(
            backend_call_path=backend_call_path,
            workflow_id=instance.instance_id,
            call_records=backend_call_records,
        )
        return {
            "instance_id": instance.instance_id,
            "status": "COMPLETED",
            "provider": getattr(self.backend, "name", self.backend.__class__.__name__),
            "agent_family": "synthetic_stress",
            "prompt_runtime_mode": self.prompt_runtime_mode,
            "plan": final_plan_text,
            "patch": "",
            "verification": final_critique_text,
            "trace_path": str(trace_path),
            "runtime_event_path": str(runtime_event_path),
            "runtime_event_summary": runtime_summary,
            "backend_call_path": str(backend_call_path),
            "backend_call_summary": backend_call_summary,
            "trace_validation": {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "summary": validation.summary,
            },
        }


class DeepResearchTracedAgentRunner(TracedAgentRunner):
    def run_instance(self, instance: WorkflowInstance) -> Dict[str, object]:
        trace_path = self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}.jsonl"
        runtime_event_path = (
            self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}_runtime.jsonl"
        )
        backend_call_path = (
            self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}_backend_calls.jsonl"
        )
        selected_files = self._select_files(instance)
        source_files = dict(sorted({**instance.readmes, **selected_files}.items()))
        self._reset_monolithic_prompt_runtime_state()

        with TraceLogger(
            trace_path,
            workflow_id=instance.instance_id,
            tenant_id=self.tenant_id,
        ) as trace:
            materializer = SegmentRuntime(
                reclaim_grace_by_role=self._reclaim_grace_by_role,
            )
            system_prompt = self._research_planner_system_prompt()
            system_state = trace.create_state(
                state_id="system_v1",
                logical_key="prompt/system",
                state_type="agent_anchor",
                size_bytes=size_bytes(system_prompt),
                token_count=approx_token_count(system_prompt),
                producer="runner",
                materialization="HBM",
                metadata=module_metadata(
                    context_module="system",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="static",
                    agent_family="deep_research",
                ),
            )
            self._register_runtime_segment(
                materializer=materializer,
                handle=system_state,
                workflow_id=instance.instance_id,
                module="system",
                role="system",
                text=system_prompt,
                materialization="HBM",
                is_shared=True,
                is_immutable=True,
                is_ephemeral=False,
                metadata={"agent_family": "deep_research"},
            )
            task_state = trace.create_state(
                state_id="task_v1",
                logical_key="prompt/task",
                state_type="conversation_history",
                size_bytes=size_bytes(instance.problem_statement),
                token_count=approx_token_count(instance.problem_statement),
                producer="runner",
                materialization="HBM",
                metadata=module_metadata(
                    context_module="task",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="task_fixed",
                    agent_family="deep_research",
                ),
            )
            self._register_runtime_segment(
                materializer=materializer,
                handle=task_state,
                workflow_id=instance.instance_id,
                module="task",
                role="task",
                text=instance.problem_statement,
                materialization="HBM",
                is_shared=True,
                is_immutable=True,
                is_ephemeral=False,
                metadata={"agent_family": "deep_research"},
            )
            source_states = self._create_text_states(
                trace=trace,
                state_type="retrieved_document",
                producer="retriever",
                logical_prefix="research_source",
                files=source_files,
                materialization="CPU",
                context_module="evidence",
                lifecycle_class="long",
                is_immutable=True,
                is_shared=True,
                is_ephemeral=False,
                update_cause="retrieval_seed",
            )
            for state, (_, contents) in zip(source_states, sorted(source_files.items())):
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=state,
                    workflow_id=instance.instance_id,
                    module="evidence",
                    role="evidence",
                    text=contents,
                    materialization="CPU",
                    is_shared=True,
                    is_immutable=True,
                    is_ephemeral=False,
                    metadata={"agent_family": "deep_research"},
                )

            live_state_ids = {system_state.state_id, task_state.state_id}
            live_state_ids.update(state.state_id for state in source_states)
            note_state_ids: List[str] = []
            current_plan_id: str | None = None
            current_draft_id: str | None = None
            current_critique_id: str | None = None
            final_plan_text = ""
            final_draft_text = ""
            final_critique_text = ""

            source_state_cycle = list(source_states) or []

            for iteration in range(1, self.max_iterations + 1):
                planner_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                ]
                planner_segments.extend(
                    self._segment(state.state_id, "evidence", role="evidence")
                    for state in source_states
                )
                planner_segments.extend(
                    self._segment(state_id, "summary", role="summary")
                    for state_id in note_state_ids
                )
                if current_draft_id is not None:
                    planner_segments.append(
                        self._segment(
                            current_draft_id,
                            "artifact",
                            role="artifact",
                            update_cause="draft_feedback",
                        )
                    )
                if current_critique_id is not None:
                    planner_segments.append(
                        self._segment(
                            current_critique_id,
                            "scratchpad",
                            role="scratchpad",
                            update_cause="critic_feedback",
                        )
                    )
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="research_planner",
                    step_name="research_planner",
                    prompt_id=f"research-planner-{iteration}",
                    system_prompt=self._research_planner_system_prompt(),
                    user_prompt=self._research_planner_user_prompt(instance, iteration),
                    segments=planner_segments,
                    metadata={"hook": "research.planner", "iteration": iteration},
                )
                plan_text = self.backend.complete(
                    step_name="research_planner",
                    system_prompt=self._research_planner_system_prompt(),
                    user_prompt=self._research_planner_user_prompt(instance, iteration),
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                plan_state = trace.create_state(
                    state_id=f"research_plan_v{iteration}",
                    logical_key="research/planner/plan",
                    state_type="plan",
                    size_bytes=size_bytes(plan_text),
                    token_count=approx_token_count(plan_text),
                    producer="research_planner",
                    parent_state_ids=(
                        [current_critique_id] if current_critique_id is not None else None
                    ),
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="plan",
                        lifecycle_class="medium",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=False,
                        update_cause="research_replan",
                        iteration=iteration,
                        agent_family="deep_research",
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=plan_state,
                    workflow_id=instance.instance_id,
                    module="plan",
                    role="plan",
                    text=plan_text,
                    materialization="HBM",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=False,
                    metadata={"agent_family": "deep_research"},
                )
                live_state_ids.add(plan_state.state_id)
                if current_plan_id is not None:
                    trace.supersede_state(
                        current_plan_id,
                        plan_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        current_plan_id,
                        consumer="research_planner",
                        metadata={"reason": "research_replan"},
                    )
                    self._release_runtime_segment(
                        materializer=materializer,
                        state_id=current_plan_id,
                    )
                    live_state_ids.discard(current_plan_id)
                current_plan_id = plan_state.state_id
                final_plan_text = plan_text

                focused_source = source_state_cycle[(iteration - 1) % len(source_state_cycle)]
                focused_source_name = focused_source.logical_key.split("/")[-1]
                reader_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_plan_id, "plan", role="plan"),
                    self._segment(focused_source.state_id, "evidence", role="evidence"),
                ]
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="research_reader",
                    step_name="research_reader",
                    prompt_id=f"research-reader-{iteration}",
                    system_prompt=self._research_reader_system_prompt(),
                    user_prompt=self._research_reader_user_prompt(
                        instance,
                        iteration,
                        focused_source_name,
                    ),
                    segments=reader_segments,
                    metadata={
                        "hook": "research.reader",
                        "iteration": iteration,
                        "source": focused_source_name,
                    },
                )
                note_text = self.backend.complete(
                    step_name="research_reader",
                    system_prompt=self._research_reader_system_prompt(),
                    user_prompt=self._research_reader_user_prompt(
                        instance,
                        iteration,
                        focused_source_name,
                    ),
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                note_state = trace.create_state(
                    state_id=f"research_note_v{iteration}",
                    logical_key=f"research/reader/note/{focused_source_name}",
                    state_type="summary",
                    size_bytes=size_bytes(note_text),
                    token_count=approx_token_count(note_text),
                    producer="research_reader",
                    parent_state_ids=[current_plan_id, focused_source.state_id],
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="summary",
                        lifecycle_class="medium",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=False,
                        update_cause="source_read",
                        iteration=iteration,
                        source=focused_source_name,
                        agent_family="deep_research",
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=note_state,
                    workflow_id=instance.instance_id,
                    module="summary",
                    role="summary",
                    text=note_text,
                    materialization="CPU",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=False,
                    metadata={"agent_family": "deep_research"},
                )
                live_state_ids.add(note_state.state_id)
                note_state_ids.append(note_state.state_id)

                writer_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_plan_id, "plan", role="plan"),
                ]
                writer_segments.extend(
                    self._segment(state_id, "summary", role="summary")
                    for state_id in note_state_ids
                )
                if current_draft_id is not None:
                    writer_segments.append(
                        self._segment(
                            current_draft_id,
                            "artifact",
                            role="artifact",
                            update_cause="draft_revision",
                        )
                    )
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="research_writer",
                    step_name="research_writer",
                    prompt_id=f"research-writer-{iteration}",
                    system_prompt=self._research_writer_system_prompt(),
                    user_prompt=self._research_writer_user_prompt(instance, iteration),
                    segments=writer_segments,
                    metadata={"hook": "research.writer", "iteration": iteration},
                )
                draft_text = self.backend.complete(
                    step_name="research_writer",
                    system_prompt=self._research_writer_system_prompt(),
                    user_prompt=self._research_writer_user_prompt(instance, iteration),
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                draft_state = trace.create_state(
                    state_id=f"research_draft_v{iteration}",
                    logical_key="research/writer/draft",
                    state_type="generated_artifact",
                    size_bytes=size_bytes(draft_text),
                    token_count=approx_token_count(draft_text),
                    producer="research_writer",
                    parent_state_ids=[current_plan_id, *note_state_ids],
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="artifact",
                        lifecycle_class="medium",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=False,
                        update_cause="draft_revision",
                        iteration=iteration,
                        agent_family="deep_research",
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=draft_state,
                    workflow_id=instance.instance_id,
                    module="artifact",
                    role="artifact",
                    text=draft_text,
                    materialization="HBM",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=False,
                    metadata={"agent_family": "deep_research"},
                )
                live_state_ids.add(draft_state.state_id)
                if current_draft_id is not None:
                    trace.supersede_state(
                        current_draft_id,
                        draft_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        current_draft_id,
                        consumer="research_writer",
                        metadata={"reason": "draft_refresh"},
                    )
                    self._release_runtime_segment(
                        materializer=materializer,
                        state_id=current_draft_id,
                    )
                    live_state_ids.discard(current_draft_id)
                current_draft_id = draft_state.state_id
                final_draft_text = draft_text

                critic_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(current_plan_id, "plan", role="plan"),
                    self._segment(current_draft_id, "artifact", role="artifact"),
                ]
                critic_segments.extend(
                    self._segment(state_id, "summary", role="summary")
                    for state_id in note_state_ids
                )
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="research_critic",
                    step_name="research_critic",
                    prompt_id=f"research-critic-{iteration}",
                    system_prompt=self._research_critic_system_prompt(),
                    user_prompt=self._research_critic_user_prompt(instance, iteration),
                    segments=critic_segments,
                    metadata={"hook": "research.critic", "iteration": iteration},
                )
                critique_text = self.backend.complete(
                    step_name="research_critic",
                    system_prompt=self._research_critic_system_prompt(),
                    user_prompt=self._research_critic_user_prompt(instance, iteration),
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                critique_state = trace.create_state(
                    state_id=f"research_critique_v{iteration}",
                    logical_key="research/critic/feedback",
                    state_type="error_diagnostic",
                    size_bytes=size_bytes(critique_text),
                    token_count=approx_token_count(critique_text),
                    producer="research_critic",
                    parent_state_ids=[current_draft_id],
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="scratchpad",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="draft_feedback",
                        iteration=iteration,
                        agent_family="deep_research",
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=critique_state,
                    workflow_id=instance.instance_id,
                    module="scratchpad",
                    role="scratchpad",
                    text=critique_text,
                    materialization="CPU",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=True,
                    metadata={"agent_family": "deep_research"},
                )
                live_state_ids.add(critique_state.state_id)
                if current_critique_id is not None:
                    trace.supersede_state(
                        current_critique_id,
                        critique_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        current_critique_id,
                        consumer="research_critic",
                        metadata={"reason": "critique_refresh"},
                    )
                    self._release_runtime_segment(
                        materializer=materializer,
                        state_id=current_critique_id,
                    )
                    live_state_ids.discard(current_critique_id)
                current_critique_id = critique_state.state_id
                final_critique_text = critique_text

            for state_id in sorted(live_state_ids):
                trace.release_state(
                    state_id,
                    consumer="workflow",
                    metadata={"reason": "workflow_end"},
                )
                self._release_runtime_segment(materializer=materializer, state_id=state_id)
            if self.prompt_runtime_mode == "monolithic":
                self._release_monolithic_prompt_runtime(materializer=materializer)

        events = load_trace_events(trace_path)
        validation = validate_trace_events(events)
        runtime_summary = self._runtime_event_summary(materializer)
        backend_call_records = self.backend.drain_call_records()
        backend_call_summary = self._backend_call_summary(backend_call_records)
        self._write_runtime_events(
            materializer=materializer,
            runtime_event_path=runtime_event_path,
            workflow_id=instance.instance_id,
            agent_family="deep_research",
        )
        self._write_backend_call_records(
            backend_call_path=backend_call_path,
            workflow_id=instance.instance_id,
            call_records=backend_call_records,
        )
        return {
            "instance_id": instance.instance_id,
            "status": "COMPLETED",
            "provider": getattr(self.backend, "name", self.backend.__class__.__name__),
            "agent_family": "deep_research",
            "prompt_runtime_mode": self.prompt_runtime_mode,
            "plan": final_plan_text,
            "patch": final_draft_text,
            "verification": final_critique_text,
            "trace_path": str(trace_path),
            "runtime_event_path": str(runtime_event_path),
            "runtime_event_summary": runtime_summary,
            "backend_call_path": str(backend_call_path),
            "backend_call_summary": backend_call_summary,
            "trace_validation": {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "summary": validation.summary,
            },
        }


class OpenDeepResearchTracedAgentRunner(DeepResearchTracedAgentRunner):
    def __init__(
        self,
        *,
        backend: ModelBackend,
        trace_dir: str | Path,
        tenant_id: str = "local",
        max_iterations: int = 2,
        max_files: int = 5,
        prompt_runtime_mode: str = "monolithic",
        open_deep_research_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            backend=backend,
            trace_dir=trace_dir,
            tenant_id=tenant_id,
            max_iterations=max_iterations,
            max_files=max_files,
            prompt_runtime_mode=prompt_runtime_mode,
        )
        self._open_deep_research_prompts = load_open_deep_research_prompt_pack(
            open_deep_research_path
        )

    def _open_deep_research_today(self) -> str:
        now = time.localtime()
        return f"{time.strftime('%a %b', now)} {now.tm_mday}, {time.strftime('%Y', now)}"

    def _open_deep_research_messages(self, instance: WorkflowInstance) -> str:
        return (
            "Human: "
            f"{instance.problem_statement}\n"
            "Assistant: Begin deep research using the provided context."
        )

    def _research_planner_system_prompt(self) -> str:
        return self._open_deep_research_prompts.lead_researcher_prompt.format(
            date=self._open_deep_research_today(),
            max_concurrent_research_units=1,
            max_researcher_iterations=self.max_iterations,
        )

    def _research_planner_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        research_brief = (
            self._open_deep_research_prompts.transform_messages_into_research_topic_prompt.format(
                messages=self._open_deep_research_messages(instance),
                date=self._open_deep_research_today(),
            )
        )
        return (
            f"{research_brief}\n\n"
            f"Supervisor iteration: {iteration}\n"
            "Use the already provided source segments to decide the next research emphasis."
        )

    def _research_reader_system_prompt(self) -> str:
        return self._open_deep_research_prompts.research_system_prompt.format(
            mcp_prompt="",
            date=self._open_deep_research_today(),
        )

    def _research_reader_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
        source_name: str,
    ) -> str:
        return (
            f"Research topic:\n{instance.problem_statement}\n\n"
            f"Focused source: {source_name}\n"
            f"Research iteration: {iteration}\n"
            "Use the provided evidence segment instead of web search, extract systems-relevant "
            "facts, and preserve concrete claims that should survive into the final report."
        )

    def _research_writer_system_prompt(self) -> str:
        return self._open_deep_research_prompts.compress_research_system_prompt.format(
            date=self._open_deep_research_today()
        )

    def _research_writer_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"{self._open_deep_research_prompts.compress_research_simple_human_message}\n\n"
            f"Research topic:\n{instance.problem_statement}\n\n"
            f"Compression iteration: {iteration}\n"
            "Clean and consolidate the currently active findings into reusable notes."
        )

    def _research_critic_system_prompt(self) -> str:
        return (
            "You are the final report generator for the open_deep_research workflow. "
            "Synthesize the active brief and accumulated notes into the next report revision."
        )

    def _research_critic_user_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return self._open_deep_research_prompts.final_report_generation_prompt.format(
            date=self._open_deep_research_today(),
            research_brief=instance.problem_statement,
            messages=self._open_deep_research_messages(instance),
            findings=(
                "Use the active plan, accumulated summary segments, and current draft segment "
                "that were provided through the runtime context."
            ),
        ) + f"\n\nReport iteration: {iteration}\n"

    def run_instance(self, instance: WorkflowInstance) -> Dict[str, object]:
        result = super().run_instance(instance)
        result["workload_source"] = "open_deep_research"
        result["open_deep_research_path"] = str(self._open_deep_research_prompts.repo_path)
        return result


class LangGraphTracedAgentRunner(TracedAgentRunner):
    def _agent_family_name(self) -> str:
        return "langgraph"

    def _result_workload_metadata(self) -> Dict[str, object]:
        return {}

    def run_instance(self, instance: WorkflowInstance) -> Dict[str, object]:
        StateGraph, START, END = _load_langgraph_symbols()

        trace_path = self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}.jsonl"
        runtime_event_path = self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}_runtime.jsonl"
        backend_call_path = self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}_backend_calls.jsonl"
        selected_files = self._select_files(instance)
        self._reset_monolithic_prompt_runtime_state()
        agent_family = self._agent_family_name()

        with TraceLogger(
            trace_path,
            workflow_id=instance.instance_id,
            tenant_id=self.tenant_id,
        ) as trace:
            materializer = SegmentRuntime(
                reclaim_grace_by_role=self._reclaim_grace_by_role,
            )
            system_prompt = self._system_prompt_text()
            system_state = trace.create_state(
                state_id="system_v1",
                logical_key="prompt/system",
                state_type="agent_anchor",
                size_bytes=size_bytes(system_prompt),
                token_count=approx_token_count(system_prompt),
                producer="runner",
                recompute_cost=1.0,
                reload_cost=0.5,
                materialization="HBM",
                metadata=module_metadata(
                    context_module="system",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="static",
                    agent_family=agent_family,
                ),
            )
            self._register_runtime_segment(
                materializer=materializer,
                handle=system_state,
                workflow_id=instance.instance_id,
                module="system",
                role="system",
                text=system_prompt,
                materialization="HBM",
                is_shared=True,
                is_immutable=True,
                is_ephemeral=False,
                metadata={"agent_family": agent_family},
            )
            task_state = trace.create_state(
                state_id="task_v1",
                logical_key="prompt/task",
                state_type="conversation_history",
                size_bytes=size_bytes(instance.problem_statement),
                token_count=approx_token_count(instance.problem_statement),
                producer="runner",
                recompute_cost=1.0,
                reload_cost=0.5,
                materialization="HBM",
                metadata=module_metadata(
                    context_module="task",
                    lifecycle_class="long",
                    is_immutable=True,
                    is_shared=True,
                    is_ephemeral=False,
                    update_cause="task_fixed",
                    agent_family=agent_family,
                ),
            )
            self._register_runtime_segment(
                materializer=materializer,
                handle=task_state,
                workflow_id=instance.instance_id,
                module="task",
                role="task",
                text=instance.problem_statement,
                materialization="HBM",
                is_shared=True,
                is_immutable=True,
                is_ephemeral=False,
                metadata={"agent_family": agent_family},
            )
            readme_states = self._create_text_states(
                trace=trace,
                state_type="retrieved_document",
                producer="retriever",
                logical_prefix="readme",
                files=instance.readmes,
                materialization="CPU",
                context_module="evidence",
                lifecycle_class="medium",
                is_immutable=True,
                is_shared=True,
                is_ephemeral=False,
                update_cause="retrieval_refresh",
            )
            doc_states = self._create_text_states(
                trace=trace,
                state_type="retrieved_document",
                producer="retriever",
                logical_prefix="retrieval",
                files=selected_files,
                materialization="CPU",
                context_module="evidence",
                lifecycle_class="short",
                is_immutable=True,
                is_shared=False,
                is_ephemeral=False,
                update_cause="retrieval_refresh",
            )
            for state, (_, contents) in zip(readme_states, sorted(instance.readmes.items())):
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=state,
                    workflow_id=instance.instance_id,
                    module="evidence",
                    role="evidence",
                    text=contents,
                    materialization="CPU",
                    is_shared=True,
                    is_immutable=True,
                    is_ephemeral=False,
                    metadata={"agent_family": agent_family},
                )
            for state, (_, contents) in zip(doc_states, sorted(selected_files.items())):
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=state,
                    workflow_id=instance.instance_id,
                    module="evidence",
                    role="evidence",
                    text=contents,
                    materialization="CPU",
                    is_shared=False,
                    is_immutable=True,
                    is_ephemeral=False,
                    metadata={"agent_family": agent_family},
                )
            live_state_ids = {system_state.state_id, task_state.state_id}
            live_state_ids.update(state.state_id for state in readme_states)
            live_state_ids.update(state.state_id for state in doc_states)

            def router_node(state: LangGraphRunnerState) -> LangGraphRunnerState:
                iteration = int(state["iteration"])
                diagnostic_state_id = state.get("diagnostic_state_id")
                router_user_prompt = (
                    self._router_instruction_prompt(
                        instance,
                        selected_files,
                        iteration,
                        diagnostic_state_id,
                    )
                    if self.prompt_runtime_mode == "segment_aware"
                    else self._router_user_prompt(
                        instance,
                        selected_files,
                        iteration,
                        diagnostic_state_id,
                    )
                )
                router_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                ]
                router_segments.extend(
                    self._segment(doc.state_id, "evidence", role="evidence")
                    for doc in readme_states
                )
                router_segments.extend(
                    self._segment(doc.state_id, "evidence", role="evidence")
                    for doc in doc_states
                )
                if diagnostic_state_id is not None:
                    router_segments.append(
                        self._segment(
                            diagnostic_state_id,
                            "scratchpad",
                            role="scratchpad",
                            update_cause="failure_feedback",
                        )
                    )
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="router",
                    step_name="router",
                    prompt_id=f"router-{iteration}",
                    system_prompt=self._router_system_prompt(),
                    user_prompt=router_user_prompt,
                    segments=router_segments,
                    metadata={"hook": "router.messages_for_llm", "iteration": iteration},
                )
                route_text = self.backend.complete(
                    step_name="router",
                    system_prompt=self._router_system_prompt(),
                    user_prompt=router_user_prompt,
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                route_state = trace.create_state(
                    state_id=f"route_v{iteration}",
                    logical_key="router/decision",
                    state_type="summary",
                    size_bytes=size_bytes(route_text),
                    token_count=approx_token_count(route_text),
                    producer="router",
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="summary",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="reroute",
                        iteration=iteration,
                        agent_family=agent_family,
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=route_state,
                    workflow_id=instance.instance_id,
                    module="summary",
                    role="router",
                    text=route_text,
                    materialization="CPU",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=True,
                    metadata={"agent_family": agent_family},
                )
                live_state_ids.add(route_state.state_id)
                current_route_id = state.get("current_route_id")
                if current_route_id is not None:
                    trace.supersede_state(current_route_id, route_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_route_id, consumer="router", metadata={"reason": "reroute"})
                    self._release_runtime_segment(materializer=materializer, state_id=current_route_id)
                    live_state_ids.discard(current_route_id)
                return {"current_route_id": route_state.state_id}

            def planner_node(state: LangGraphRunnerState) -> LangGraphRunnerState:
                iteration = int(state["iteration"])
                diagnostic_state_id = state.get("diagnostic_state_id")
                planner_user_prompt = self._planner_user_prompt(
                    instance,
                    selected_files,
                    iteration,
                    diagnostic_state_id,
                )
                planner_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(str(state["current_route_id"]), "summary", role="router"),
                ]
                planner_segments.extend(
                    self._segment(doc.state_id, "evidence", role="evidence")
                    for doc in readme_states
                )
                planner_segments.extend(
                    self._segment(doc.state_id, "evidence", role="evidence")
                    for doc in doc_states
                )
                if diagnostic_state_id is not None:
                    planner_segments.append(
                        self._segment(
                            diagnostic_state_id,
                            "scratchpad",
                            role="scratchpad",
                            update_cause="previous_failure",
                        )
                    )
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="planner",
                    step_name="planner",
                    prompt_id=f"planner-{iteration}",
                    system_prompt=self._planner_system_prompt(),
                    user_prompt=planner_user_prompt,
                    segments=planner_segments,
                    metadata={"hook": "planner.messages_for_llm", "iteration": iteration},
                )
                plan_text = self.backend.complete(
                    step_name="planner",
                    system_prompt=self._planner_system_prompt(),
                    user_prompt=planner_user_prompt,
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                parent_ids = [str(state["current_route_id"])]
                if diagnostic_state_id is not None:
                    parent_ids.append(diagnostic_state_id)
                plan_state = trace.create_state(
                    state_id=f"plan_v{iteration}",
                    logical_key="planner/plan",
                    state_type="plan",
                    size_bytes=size_bytes(plan_text),
                    token_count=approx_token_count(plan_text),
                    producer="planner",
                    parent_state_ids=parent_ids,
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="plan",
                        lifecycle_class="medium",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=False,
                        update_cause="replan",
                        iteration=iteration,
                        agent_family=agent_family,
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=plan_state,
                    workflow_id=instance.instance_id,
                    module="plan",
                    role="plan",
                    text=plan_text,
                    materialization="HBM",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=False,
                    metadata={"agent_family": agent_family},
                )
                live_state_ids.add(plan_state.state_id)
                current_plan_id = state.get("current_plan_id")
                if current_plan_id is not None:
                    trace.supersede_state(current_plan_id, plan_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_plan_id, consumer="planner", metadata={"reason": "replan"})
                    self._release_runtime_segment(materializer=materializer, state_id=current_plan_id)
                    live_state_ids.discard(current_plan_id)
                return {
                    "current_plan_id": plan_state.state_id,
                    "final_plan_text": plan_text,
                }

            def coder_node(state: LangGraphRunnerState) -> LangGraphRunnerState:
                iteration = int(state["iteration"])
                plan_text = str(state["final_plan_text"])
                coder_user_prompt = (
                    self._coder_instruction_prompt(instance, iteration)
                    if self.prompt_runtime_mode == "segment_aware"
                    else self._coder_user_prompt(instance, selected_files, plan_text, iteration)
                )
                coder_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(str(state["current_route_id"]), "summary", role="router"),
                    self._segment(str(state["current_plan_id"]), "plan", role="plan", update_cause="plan_update"),
                ]
                coder_segments.extend(
                    self._segment(doc.state_id, "evidence", role="evidence")
                    for doc in readme_states
                )
                coder_segments.extend(
                    self._segment(doc.state_id, "evidence", role="evidence")
                    for doc in doc_states
                )
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="coder",
                    step_name="coder",
                    prompt_id=f"coder-{iteration}",
                    system_prompt=self._coder_system_prompt(),
                    user_prompt=coder_user_prompt,
                    segments=coder_segments,
                    metadata={"hook": "coder.messages_for_llm", "iteration": iteration},
                )
                patch_text = self.backend.complete(
                    step_name="coder",
                    system_prompt=self._coder_system_prompt(),
                    user_prompt=coder_user_prompt,
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                patch_parent_ids = [str(state["current_route_id"]), str(state["current_plan_id"])]
                patch_parent_ids.extend(doc.state_id for doc in doc_states)
                patch_state = trace.create_state(
                    state_id=f"patch_v{iteration}",
                    logical_key="coder/patch",
                    state_type="generated_artifact",
                    size_bytes=size_bytes(patch_text),
                    token_count=approx_token_count(patch_text),
                    producer="coder",
                    parent_state_ids=patch_parent_ids,
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="artifact",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="repatch",
                        iteration=iteration,
                        agent_family=agent_family,
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=patch_state,
                    workflow_id=instance.instance_id,
                    module="artifact",
                    role="artifact",
                    text=patch_text,
                    materialization="HBM",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=True,
                    metadata={"agent_family": agent_family},
                )
                live_state_ids.add(patch_state.state_id)
                current_patch_id = state.get("current_patch_id")
                if current_patch_id is not None:
                    trace.supersede_state(current_patch_id, patch_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_patch_id, consumer="coder", metadata={"reason": "repatch"})
                    self._release_runtime_segment(materializer=materializer, state_id=current_patch_id)
                    live_state_ids.discard(current_patch_id)
                return {
                    "current_patch_id": patch_state.state_id,
                    "final_patch_text": patch_text,
                }

            def reviewer_node(state: LangGraphRunnerState) -> LangGraphRunnerState:
                iteration = int(state["iteration"])
                plan_text = str(state["final_plan_text"])
                patch_text = str(state["final_patch_text"])
                reviewer_user_prompt = (
                    self._reviewer_instruction_prompt(instance, iteration)
                    if self.prompt_runtime_mode == "segment_aware"
                    else self._reviewer_user_prompt(instance, plan_text, patch_text, iteration)
                )
                reviewer_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(str(state["current_route_id"]), "summary", role="router"),
                    self._segment(str(state["current_plan_id"]), "plan", role="plan", update_cause="plan_update"),
                    self._segment(str(state["current_patch_id"]), "artifact", role="artifact", update_cause="patch_update"),
                ]
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="reviewer",
                    step_name="reviewer",
                    prompt_id=f"reviewer-{iteration}",
                    system_prompt=self._reviewer_system_prompt(),
                    user_prompt=reviewer_user_prompt,
                    segments=reviewer_segments,
                    metadata={"hook": "reviewer.messages_for_llm", "iteration": iteration},
                )
                review_text = self.backend.complete(
                    step_name="reviewer",
                    system_prompt=self._reviewer_system_prompt(),
                    user_prompt=reviewer_user_prompt,
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                review_state = trace.create_state(
                    state_id=f"review_v{iteration}",
                    logical_key="reviewer/notes",
                    state_type="summary",
                    size_bytes=size_bytes(review_text),
                    token_count=approx_token_count(review_text),
                    producer="reviewer",
                    parent_state_ids=[str(state["current_patch_id"])],
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="summary",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="review_update",
                        iteration=iteration,
                        agent_family=agent_family,
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=review_state,
                    workflow_id=instance.instance_id,
                    module="summary",
                    role="review",
                    text=review_text,
                    materialization="CPU",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=True,
                    metadata={"agent_family": agent_family},
                )
                live_state_ids.add(review_state.state_id)
                current_review_id = state.get("current_review_id")
                if current_review_id is not None:
                    trace.supersede_state(current_review_id, review_state.state_id, metadata={"iteration": iteration})
                    trace.release_state(current_review_id, consumer="reviewer", metadata={"reason": "rereview"})
                    self._release_runtime_segment(materializer=materializer, state_id=current_review_id)
                    live_state_ids.discard(current_review_id)
                return {"current_review_id": review_state.state_id}

            def tester_node(state: LangGraphRunnerState) -> LangGraphRunnerState:
                iteration = int(state["iteration"])
                plan_text = str(state["final_plan_text"])
                patch_text = str(state["final_patch_text"])
                tester_user_prompt = (
                    self._tester_instruction_prompt(instance, iteration)
                    if self.prompt_runtime_mode == "segment_aware"
                    else self._tester_user_prompt(instance, plan_text, patch_text, iteration)
                )
                tester_segments = [
                    self._segment(system_state.state_id, "system", role="system"),
                    self._segment(task_state.state_id, "task", role="task"),
                    self._segment(str(state["current_patch_id"]), "artifact", role="artifact", update_cause="patch_update"),
                    self._segment(str(state["current_review_id"]), "summary", role="review"),
                ]
                prompt_mode, prompt_group, segment_request, snapshot = self._prepare_langgraph_prompt_runtime(
                    materializer=materializer,
                    trace=trace,
                    workflow_id=instance.instance_id,
                    consumer="tester",
                    step_name="tester",
                    prompt_id=f"tester-{iteration}",
                    system_prompt=self._tester_system_prompt(),
                    user_prompt=tester_user_prompt,
                    segments=tester_segments,
                    metadata={"hook": "tester.messages_for_llm", "iteration": iteration},
                )
                verdict_text = self.backend.complete(
                    step_name="tester",
                    system_prompt=self._tester_system_prompt(),
                    user_prompt=tester_user_prompt,
                    iteration=iteration,
                    instance=instance,
                    prompt_mode=prompt_mode,
                    prompt_group=prompt_group,
                    segment_request=segment_request,
                    materializer_snapshot=snapshot,
                )
                verification_state = trace.create_state(
                    state_id=f"verification_v{iteration}",
                    logical_key="tester/result",
                    state_type="verification_result",
                    size_bytes=size_bytes(verdict_text),
                    token_count=approx_token_count(verdict_text),
                    producer="tester",
                    parent_state_ids=[str(state["current_patch_id"]), str(state["current_review_id"])],
                    materialization="HBM",
                    metadata=module_metadata(
                        context_module="verification",
                        lifecycle_class="short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="retest",
                        iteration=iteration,
                        agent_family=agent_family,
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=verification_state,
                    workflow_id=instance.instance_id,
                    module="verification",
                    role="verification",
                    text=verdict_text,
                    materialization="HBM",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=True,
                    metadata={"agent_family": agent_family},
                )
                live_state_ids.add(verification_state.state_id)
                current_verification_id = state.get("current_verification_id")
                if current_verification_id is not None:
                    trace.supersede_state(
                        current_verification_id,
                        verification_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        current_verification_id,
                        consumer="tester",
                        metadata={"reason": "retest"},
                    )
                    self._release_runtime_segment(materializer=materializer, state_id=current_verification_id)
                    live_state_ids.discard(current_verification_id)

                status = self._classify_verdict(verdict_text)
                output: LangGraphRunnerState = {
                    "current_verification_id": verification_state.state_id,
                    "final_verdict_text": verdict_text,
                    "status": status,
                    "continue_loop": False,
                }
                if status == "RESOLVED":
                    return output
                if iteration >= self.max_iterations:
                    return output

                diagnostic_text = self._diagnostic_from_verdict(verdict_text)
                diagnostic_state = trace.create_state(
                    state_id=f"diagnostic_v{iteration}",
                    logical_key="tester/diagnostic",
                    state_type="error_diagnostic",
                    size_bytes=size_bytes(diagnostic_text),
                    token_count=approx_token_count(diagnostic_text),
                    producer="tester",
                    parent_state_ids=[verification_state.state_id],
                    materialization="CPU",
                    metadata=module_metadata(
                        context_module="scratchpad",
                        lifecycle_class="very_short",
                        is_immutable=False,
                        is_shared=False,
                        is_ephemeral=True,
                        update_cause="failure_feedback",
                        iteration=iteration,
                        agent_family=agent_family,
                    ),
                )
                self._register_runtime_segment(
                    materializer=materializer,
                    handle=diagnostic_state,
                    workflow_id=instance.instance_id,
                    module="scratchpad",
                    role="scratchpad",
                    text=diagnostic_text,
                    materialization="CPU",
                    is_shared=False,
                    is_immutable=False,
                    is_ephemeral=True,
                    metadata={"agent_family": agent_family},
                )
                live_state_ids.add(diagnostic_state.state_id)
                previous_diagnostic_state_id = state.get("diagnostic_state_id")
                if previous_diagnostic_state_id is not None:
                    trace.supersede_state(
                        previous_diagnostic_state_id,
                        diagnostic_state.state_id,
                        metadata={"iteration": iteration},
                    )
                    trace.release_state(
                        previous_diagnostic_state_id,
                        consumer="router",
                        metadata={"reason": "new_diagnostic"},
                    )
                    self._release_runtime_segment(
                        materializer=materializer,
                        state_id=previous_diagnostic_state_id,
                    )
                    live_state_ids.discard(previous_diagnostic_state_id)
                output["diagnostic_state_id"] = diagnostic_state.state_id
                output["continue_loop"] = True
                output["iteration"] = iteration + 1
                return output

            graph = StateGraph(LangGraphRunnerState)
            graph.add_node("router", router_node)
            graph.add_node("planner", planner_node)
            graph.add_node("coder", coder_node)
            graph.add_node("reviewer", reviewer_node)
            graph.add_node("tester", tester_node)
            graph.add_edge(START, "router")
            graph.add_edge("router", "planner")
            graph.add_edge("planner", "coder")
            graph.add_edge("coder", "reviewer")
            graph.add_edge("reviewer", "tester")
            graph.add_conditional_edges(
                "tester",
                lambda state: "router" if state.get("continue_loop") else END,
                {"router": "router", END: END},
            )
            app = graph.compile()
            final_state = app.invoke(
                {
                    "iteration": 1,
                    "status": "UNRESOLVED",
                    "final_plan_text": "",
                    "final_patch_text": "",
                    "final_verdict_text": "",
                    "continue_loop": False,
                },
                {"recursion_limit": max(10, self.max_iterations * 8)},
            )

            for state_id in sorted(live_state_ids):
                trace.release_state(state_id, consumer="workflow", metadata={"reason": "workflow_end"})
                self._release_runtime_segment(materializer=materializer, state_id=state_id)
            if self.prompt_runtime_mode == "monolithic":
                self._release_monolithic_prompt_runtime(materializer=materializer)

        events = load_trace_events(trace_path)
        validation = validate_trace_events(events)
        runtime_summary = self._runtime_event_summary(materializer)
        backend_call_records = self.backend.drain_call_records()
        backend_call_summary = self._backend_call_summary(backend_call_records)
        self._write_runtime_events(
            materializer=materializer,
            runtime_event_path=runtime_event_path,
            workflow_id=instance.instance_id,
            agent_family=agent_family,
        )
        self._write_backend_call_records(
            backend_call_path=backend_call_path,
            workflow_id=instance.instance_id,
            call_records=backend_call_records,
        )
        return {
            "instance_id": instance.instance_id,
            "status": str(final_state.get("status", "UNRESOLVED")),
            "provider": getattr(self.backend, "name", self.backend.__class__.__name__),
            "agent_family": agent_family,
            "prompt_runtime_mode": self.prompt_runtime_mode,
            "plan": str(final_state.get("final_plan_text", "")),
            "patch": str(final_state.get("final_patch_text", "")),
            "verification": str(final_state.get("final_verdict_text", "")),
            "trace_path": str(trace_path),
            "runtime_event_path": str(runtime_event_path),
            "runtime_event_summary": runtime_summary,
            "backend_call_path": str(backend_call_path),
            "backend_call_summary": backend_call_summary,
            "trace_validation": {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "summary": validation.summary,
            },
            **self._result_workload_metadata(),
        }

    def _router_system_prompt(self) -> str:
        return (
            "You are a routing agent. Decide whether the issue needs plan-then-patch "
            "or direct patching, and briefly justify the route."
        )

    def _router_user_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            f"Previous failure context: {diagnostic_state_id}\n"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            f"{diagnostic_text}"
            f"Candidate files:\n"
            + "\n".join(f"- {path}" for path in sorted(selected_files))
        )

    def _router_instruction_prompt(
        self,
        instance: WorkflowInstance,
        selected_files: Mapping[str, str],
        iteration: int,
        diagnostic_state_id: str | None,
    ) -> str:
        diagnostic_text = (
            f"Previous failure context segment: {diagnostic_state_id}\n"
            if diagnostic_state_id is not None
            else ""
        )
        return (
            f"Iteration: {iteration}\n"
            "Route the issue using the provided context segments.\n"
            f"{diagnostic_text}"
            "Candidate files:\n"
            + "\n".join(f"- {path}" for path in sorted(selected_files))
        )

    def _reviewer_system_prompt(self) -> str:
        return (
            "You are a review agent. Inspect the proposed patch, identify residual risks, "
            "and summarize whether it is ready for verification."
        )

    def _reviewer_user_prompt(
        self,
        instance: WorkflowInstance,
        plan_text: str,
        patch_text: str,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            f"Issue:\n{instance.problem_statement}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Patch under review:\n{patch_text}\n"
        )

    def _reviewer_instruction_prompt(
        self,
        instance: WorkflowInstance,
        iteration: int,
    ) -> str:
        return (
            f"Iteration: {iteration}\n"
            "Review the candidate patch using the provided context segments and decide "
            "whether it is ready for verification."
        )


class OpenSWEAgentTracedAgentRunner(
    _OpenSWEAgentPromptBackedMixin, LangGraphTracedAgentRunner
):
    pass


class OpenHandsTracedAgentRunner(
    _OpenHandsPromptBackedMixin, LangGraphTracedAgentRunner
):
    pass
