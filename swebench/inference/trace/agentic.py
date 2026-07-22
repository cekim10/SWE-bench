from __future__ import annotations

from collections import Counter
import json
import importlib.util
import math
import os
import re
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


class StubModelBackend:
    name = "stub"

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
        if step_name == "planner":
            return (
                f"Plan v{iteration}:\n"
                "1. Inspect `safe_ratio` and the failing test.\n"
                "2. Add zero-divisor handling.\n"
                "3. Re-run verification.\n"
            )
        if step_name == "router":
            return (
                f"Route v{iteration}: plan_then_patch\n"
                "Reason: retrieved evidence suggests a localized source fix plus lightweight verification.\n"
            )
        if step_name == "coder":
            return (
                "--- a/src/math_utils.py\n"
                "+++ b/src/math_utils.py\n"
                "@@ -1,2 +1,4 @@\n"
                " def safe_ratio(a, b):\n"
                "-    return a / b\n"
                "+    if b == 0:\n"
                "+        return 0\n"
                "+    return a / b\n"
            )
        if step_name == "tester":
            if iteration == 1:
                return (
                    "FAIL\n"
                    "The patch handles zero divisors, but you should verify edge cases and restate the intended invariant."
                )
            return "PASS\nThe patch matches the stated requirement and the targeted test should pass."
        if step_name == "reviewer":
            if iteration == 1:
                return (
                    "Review v1:\n"
                    "- Guarding the zero divisor looks correct.\n"
                    "- Verification should explicitly confirm the zero-divisor contract.\n"
                )
            return (
                "Review v2:\n"
                "- Patch matches the stated invariant.\n"
                "- No additional changes required before verification.\n"
        )
        raise ValueError(f"unknown step_name {step_name!r}")

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
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
        return self.adapter.complete(request)

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
        materializer.register_segment(segment)

    def _release_runtime_segment(
        self,
        *,
        materializer: SegmentRuntime,
        state_id: str | None,
    ) -> None:
        if state_id is None:
            return
        materializer.release_segment(state_id)

    def _prepare_segmented_prompt(
        self,
        *,
        materializer: SegmentRuntime,
        trace: TraceLogger,
        workflow_id: str,
        consumer: str,
        step_name: str,
        prompt_id: str,
        segments: Sequence[Mapping[str, object]],
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[ContextSegmentGroup, SegmentedGenerationRequest, RuntimeResidencySnapshot]:
        group = materializer.build_group(
            group_id=prompt_id,
            workflow_id=workflow_id,
            consumer=consumer,
            ordered_segment_ids=[str(segment["state_id"]) for segment in segments],
            prompt_id=prompt_id,
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
            },
        )
        return group, materializer.build_vllm_request(group), snapshot

    def _prepare_langgraph_prompt_runtime(
        self,
        *,
        materializer: SegmentRuntime,
        trace: TraceLogger,
        workflow_id: str,
        consumer: str,
        step_name: str,
        prompt_id: str,
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
                segments=segments,
                metadata=metadata,
            )
            return "segment_aware", group, request, snapshot

        trace.log_prompt_segments(
            consumer=consumer,
            prompt_id=prompt_id,
            segments=segments,
            metadata={
                **dict(metadata or {}),
                "runtime_group_id": prompt_id,
                "runtime_prompt_mode": "monolithic",
                "runtime_model_id": self.backend.runtime_model_id,
                "runtime_tokenizer_id": self.backend.runtime_tokenizer_id,
            },
        )
        return "monolithic", None, None, None

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
                    "agent_family": agent_family,
                    "provider": getattr(self.backend, "name", self.backend.__class__.__name__),
                }
                handle.write(json.dumps(record) + "\n")

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


class LangGraphTracedAgentRunner(TracedAgentRunner):
    def run_instance(self, instance: WorkflowInstance) -> Dict[str, object]:
        StateGraph, START, END = _load_langgraph_symbols()

        trace_path = self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}.jsonl"
        runtime_event_path = self.trace_dir / f"{sanitize_state_suffix(instance.instance_id)}_runtime.jsonl"
        selected_files = self._select_files(instance)

        with TraceLogger(
            trace_path,
            workflow_id=instance.instance_id,
            tenant_id=self.tenant_id,
        ) as trace:
            materializer = SegmentRuntime()
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
                    agent_family="langgraph",
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
                metadata={"agent_family": "langgraph"},
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
                    agent_family="langgraph",
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
                metadata={"agent_family": "langgraph"},
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
                    metadata={"agent_family": "langgraph"},
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
                    metadata={"agent_family": "langgraph"},
                )
            live_state_ids = {system_state.state_id, task_state.state_id}
            live_state_ids.update(state.state_id for state in readme_states)
            live_state_ids.update(state.state_id for state in doc_states)

            def router_node(state: LangGraphRunnerState) -> LangGraphRunnerState:
                iteration = int(state["iteration"])
                diagnostic_state_id = state.get("diagnostic_state_id")
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
                    segments=router_segments,
                    metadata={"hook": "router.messages_for_llm", "iteration": iteration},
                )
                route_text = self.backend.complete(
                    step_name="router",
                    system_prompt=self._router_system_prompt(),
                    user_prompt=self._router_user_prompt(
                        instance,
                        selected_files,
                        iteration,
                        diagnostic_state_id,
                    ),
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
                        agent_family="langgraph",
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
                    metadata={"agent_family": "langgraph"},
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
                    segments=planner_segments,
                    metadata={"hook": "planner.messages_for_llm", "iteration": iteration},
                )
                plan_text = self.backend.complete(
                    step_name="planner",
                    system_prompt=self._planner_system_prompt(),
                    user_prompt=self._planner_user_prompt(instance, selected_files, iteration, diagnostic_state_id),
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
                        agent_family="langgraph",
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
                    metadata={"agent_family": "langgraph"},
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
                    segments=coder_segments,
                    metadata={"hook": "coder.messages_for_llm", "iteration": iteration},
                )
                patch_text = self.backend.complete(
                    step_name="coder",
                    system_prompt=self._coder_system_prompt(),
                    user_prompt=self._coder_user_prompt(instance, selected_files, plan_text, iteration),
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
                        agent_family="langgraph",
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
                    metadata={"agent_family": "langgraph"},
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
                    segments=reviewer_segments,
                    metadata={"hook": "reviewer.messages_for_llm", "iteration": iteration},
                )
                review_text = self.backend.complete(
                    step_name="reviewer",
                    system_prompt=self._reviewer_system_prompt(),
                    user_prompt=self._reviewer_user_prompt(instance, plan_text, patch_text, iteration),
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
                        agent_family="langgraph",
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
                    metadata={"agent_family": "langgraph"},
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
                    segments=tester_segments,
                    metadata={"hook": "tester.messages_for_llm", "iteration": iteration},
                )
                verdict_text = self.backend.complete(
                    step_name="tester",
                    system_prompt=self._tester_system_prompt(),
                    user_prompt=self._tester_user_prompt(instance, plan_text, patch_text, iteration),
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
                        agent_family="langgraph",
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
                    metadata={"agent_family": "langgraph"},
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
                        agent_family="langgraph",
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
                    metadata={"agent_family": "langgraph"},
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

        events = load_trace_events(trace_path)
        validation = validate_trace_events(events)
        runtime_summary = self._runtime_event_summary(materializer)
        self._write_runtime_events(
            materializer=materializer,
            runtime_event_path=runtime_event_path,
            workflow_id=instance.instance_id,
            agent_family="langgraph",
        )
        return {
            "instance_id": instance.instance_id,
            "status": str(final_state.get("status", "UNRESOLVED")),
            "provider": getattr(self.backend, "name", self.backend.__class__.__name__),
            "agent_family": "langgraph",
            "prompt_runtime_mode": self.prompt_runtime_mode,
            "plan": str(final_state.get("final_plan_text", "")),
            "patch": str(final_state.get("final_patch_text", "")),
            "verification": str(final_state.get("final_verdict_text", "")),
            "trace_path": str(trace_path),
            "runtime_event_path": str(runtime_event_path),
            "runtime_event_summary": runtime_summary,
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
