#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

from swebench.inference.runtime.segment_materializer import (
    ContextSegment,
    ExecutionContextKey,
    SegmentIdentity,
    SegmentRole,
    SegmentRuntime,
)
from swebench.inference.runtime.vllm_adapter import (
    OpenAICompatibleVLLMAdapter,
    VLLMBackendRequest,
    VLLMCompletionResult,
)


class APCProbeAdapter(Protocol):
    def complete_with_details(self, request: VLLMBackendRequest) -> VLLMCompletionResult:
        ...


@dataclass(frozen=True)
class APCProbeCase:
    name: str
    segment_ids: tuple[str, ...]
    prefix_segment_ids: tuple[str, ...]
    prefix_digest: str
    request: VLLMBackendRequest
    execution_context_digests: Mapping[str, str]


@dataclass(frozen=True)
class APCProbeObservation:
    name: str
    segment_ids: tuple[str, ...]
    prefix_segment_ids: tuple[str, ...]
    prefix_digest: str
    message_digest: str
    duration_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_preview: str
    execution_context_digests: Mapping[str, str]


def _make_segment(
    *,
    state_id: str,
    logical_key: str,
    module: str,
    version: int,
    role: SegmentRole,
    text: str,
) -> ContextSegment:
    return ContextSegment(
        state_id=state_id,
        identity=SegmentIdentity(logical_key=logical_key, module=module),
        version=version,
        role=role,
        text=text,
        size_bytes=len(text.encode("utf-8")),
        token_count=max(1, len(text)),
        metadata={"role": role.value},
    )


def _prefix_digest_for_segments(segments: Sequence[ContextSegment]) -> str:
    prefix_messages = [
        {
            "segment_id": segment.segment_id,
            "role": segment.role.value,
            "text": segment.text,
        }
        for segment in segments
    ]
    return sha256(
        json.dumps(prefix_messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_probe_cases(*, model: str, max_tokens: int = 16) -> list[APCProbeCase]:
    runtime = SegmentRuntime()
    system = _make_segment(
        state_id="system-1",
        logical_key="prompt/system",
        module="system",
        version=1,
        role=SegmentRole.SYSTEM,
        text="You are a precise code-repair assistant.",
    )
    task = _make_segment(
        state_id="task-1",
        logical_key="prompt/task",
        module="task",
        version=1,
        role=SegmentRole.TASK,
        text="Fix the bug and keep the patch minimal.",
    )
    plan = _make_segment(
        state_id="plan-1",
        logical_key="planner/plan",
        module="plan",
        version=1,
        role=SegmentRole.PLAN,
        text="Plan:\n1. Inspect failure.\n2. Patch the bug.\n3. Re-run verification.",
    )
    evidence = _make_segment(
        state_id="evidence-1",
        logical_key="retrieval/evidence",
        module="evidence",
        version=1,
        role=SegmentRole.EVIDENCE,
        text="Evidence:\nThe failing branch divides by zero when the denominator is empty.",
    )
    query_a = _make_segment(
        state_id="query-a",
        logical_key="prompt/query",
        module="scratchpad",
        version=1,
        role=SegmentRole.SCRATCH,
        text="Return only the patch that fixes the denominator guard.",
    )
    query_b = _make_segment(
        state_id="query-b",
        logical_key="prompt/query",
        module="scratchpad",
        version=2,
        role=SegmentRole.SCRATCH,
        text="Return the patch and one sentence explaining the fix.",
    )

    for segment in (system, task, plan, evidence, query_a, query_b):
        runtime.register(segment)

    case_specs = [
        ("cold_same_prompt", [system, task, plan, evidence, query_a]),
        ("warm_same_prompt", [system, task, plan, evidence, query_a]),
        ("shared_prefix_variant", [system, task, plan, evidence, query_b]),
    ]

    cases: list[APCProbeCase] = []
    for name, segments in case_specs:
        group = runtime.build_group(
            group_id=name,
            workflow_id="apc-probe",
            consumer="apc-validator",
            ordered_segment_ids=[segment.segment_id for segment in segments],
            prompt_id=name,
        )
        request = runtime.build_vllm_request(group)
        execution_context_digests = {
            segment.segment_id: runtime.execution_context_for_group(
                group=group,
                state_id=segment.segment_id,
                model_id=model,
                tokenizer_id=model,
                inference_config={"temperature": 0.0, "max_tokens": max_tokens},
            ).digest
            for segment in segments
        }
        prefix_segments = tuple(segments[:-1])
        cases.append(
            APCProbeCase(
                name=name,
                segment_ids=tuple(segment.segment_id for segment in segments),
                prefix_segment_ids=tuple(segment.segment_id for segment in prefix_segments),
                prefix_digest=_prefix_digest_for_segments(prefix_segments),
                request=VLLMBackendRequest(
                    model=model,
                    system_prompt="",
                    user_prompt="",
                    temperature=0.0,
                    max_tokens=max_tokens,
                    prompt_mode="segment_aware",
                    segment_request=request,
                    extra_body={"probe_case": name},
                ),
                execution_context_digests=execution_context_digests,
            )
        )
    return cases


def run_probe_cases(
    *,
    adapter: APCProbeAdapter,
    cases: Iterable[APCProbeCase],
) -> dict[str, object]:
    observations: list[APCProbeObservation] = []
    for case in cases:
        result = adapter.complete_with_details(case.request)
        observations.append(
            APCProbeObservation(
                name=case.name,
                segment_ids=case.segment_ids,
                prefix_segment_ids=case.prefix_segment_ids,
                prefix_digest=case.prefix_digest,
                message_digest=result.message_digest,
                duration_ms=result.duration_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                response_preview=result.text[:160],
                execution_context_digests=dict(case.execution_context_digests),
            )
        )

    by_name = {observation.name: observation for observation in observations}
    cold = by_name["cold_same_prompt"]
    warm = by_name["warm_same_prompt"]
    sibling = by_name["shared_prefix_variant"]
    return {
        "observations": [asdict(observation) for observation in observations],
        "summary": {
            "same_prompt_message_match": cold.message_digest == warm.message_digest,
            "shared_prefix_digest_match": (
                cold.prefix_digest == warm.prefix_digest == sibling.prefix_digest
            ),
            "same_prompt_duration_ratio": (
                warm.duration_ms / cold.duration_ms if cold.duration_ms else None
            ),
            "shared_prefix_variant_duration_ratio": (
                sibling.duration_ms / cold.duration_ms if cold.duration_ms else None
            ),
            "observed_same_prompt_speedup": warm.duration_ms < cold.duration_ms,
            "observed_shared_prefix_speedup": sibling.duration_ms < cold.duration_ms,
        },
    }


def parse_args():
    parser = ArgumentParser(
        description="Run a minimal real-vLLM APC validation path with deterministic segmented prompt assembly.",
    )
    parser.add_argument("--model", required=True, help="Model served by the vLLM server.")
    parser.add_argument(
        "--output_path",
        required=True,
        help="JSON path where the APC probe report will be written.",
    )
    parser.add_argument(
        "--base_url",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible vLLM base URL.",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
        help="API key for the vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16,
        help="Small completion length to keep the APC probe cheap.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum SDK retries for the OpenAI-compatible client.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = OpenAICompatibleVLLMAdapter(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    cases = build_probe_cases(model=args.model, max_tokens=args.max_tokens)
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "probe_case_count": len(cases),
        **run_probe_cases(adapter=adapter, cases=cases),
        "limitations": [
            "This probe validates deterministic segmented prompt assembly and observed APC behavior through the OpenAI-compatible vLLM interface.",
            "It does not expose native segment-level KV handles or engine-internal reclaim operations.",
        ],
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
