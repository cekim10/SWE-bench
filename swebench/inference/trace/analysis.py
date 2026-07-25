from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


DEFAULT_MODULE_BY_STATE_TYPE = {
    "agent_anchor": "system",
    "conversation_history": "history",
    "plan": "plan",
    "tool_input": "tool_input",
    "tool_output": "tool_output",
    "retrieved_document": "evidence",
    "generated_artifact": "artifact",
    "error_diagnostic": "scratchpad",
    "summary": "summary",
    "verification_result": "verification",
    "scratch": "scratchpad",
}


@dataclass
class StateRecord:
    state_id: str
    logical_key: str
    state_type: str
    workflow_id: str
    created_ts: int
    size_bytes: int
    token_count: int
    version: int
    module: str
    lifecycle_class: str
    is_immutable: bool
    is_shared: bool
    is_ephemeral: bool
    update_cause: str
    recompute_cost: float
    reload_cost: float
    materialization: str | None
    metadata: Dict[str, object] = field(default_factory=dict)
    read_ts: List[int] = field(default_factory=list)
    read_prompt_ids: List[str] = field(default_factory=list)
    consumers: set[str] = field(default_factory=set)
    superseded_ts: int | None = None
    released_ts: int | None = None

    def end_ts(self, trace_end_ts: int) -> int:
        candidates = [trace_end_ts]
        if self.superseded_ts is not None:
            candidates.append(self.superseded_ts)
        if self.released_ts is not None:
            candidates.append(self.released_ts)
        return min(candidates)

    def lifetime(self, trace_end_ts: int) -> int:
        return max(0, self.end_ts(trace_end_ts) - self.created_ts)

    def last_read_ts(self) -> int | None:
        return self.read_ts[-1] if self.read_ts else None


@dataclass
class PromptCall:
    workflow_id: str
    consumer: str
    prompt_id: str
    ts: int
    state_ids: List[str] = field(default_factory=list)
    segment_modules: List[str] = field(default_factory=list)


def load_trace_events(path: str | Path) -> List[Dict[str, object]]:
    path = Path(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_run_output_records(path: str | Path) -> List[Dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_runtime_events(path: str | Path) -> List[Dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_backend_call_records(path: str | Path) -> List[Dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_run_output(path: str | Path) -> Dict[str, object]:
    records = load_run_output_records(path)
    provider_counts: Dict[str, int] = defaultdict(int)
    valid_trace_count = 0
    invalid_instance_ids = []
    missing_trace_path_count = 0
    runtime_event_path_count = 0
    backend_call_path_count = 0
    for record in records:
        provider = str(record.get("provider", "unknown"))
        provider_counts[provider] += 1
        validation = record.get("trace_validation") or {}
        if validation.get("is_valid", False):
            valid_trace_count += 1
        else:
            invalid_instance_ids.append(str(record.get("instance_id", "unknown")))
        if not record.get("trace_path"):
            missing_trace_path_count += 1
        if record.get("runtime_event_path"):
            runtime_event_path_count += 1
        if record.get("backend_call_path"):
            backend_call_path_count += 1

    real_backend_trace_count = sum(
        count for provider, count in provider_counts.items() if provider != "stub"
    )
    return {
        "record_count": len(records),
        "provider_counts": dict(sorted(provider_counts.items())),
        "valid_trace_count": valid_trace_count,
        "invalid_instance_ids": invalid_instance_ids,
        "missing_trace_path_count": missing_trace_path_count,
        "runtime_event_path_count": runtime_event_path_count,
        "backend_call_path_count": backend_call_path_count,
        "real_backend_trace_count": real_backend_trace_count,
        "stub_trace_count": provider_counts.get("stub", 0),
    }


def discover_trace_paths(
    *,
    trace_dir: str | Path | None = None,
    trace_paths: Sequence[str] | None = None,
    run_output_path: str | Path | None = None,
) -> List[Path]:
    discovered: List[Path] = []
    if trace_dir is not None:
        discovered.extend(sorted(Path(trace_dir).glob("*.jsonl")))
    if trace_paths is not None:
        discovered.extend(Path(path) for path in trace_paths)
    if run_output_path is not None:
        for record in load_run_output_records(run_output_path):
            trace_path = record.get("trace_path")
            if trace_path:
                discovered.append(Path(str(trace_path)))
    deduped = []
    seen = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def analyze_trace_events(events: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    states: Dict[str, StateRecord] = {}
    prompt_calls: Dict[tuple[str, str, str], PromptCall] = {}
    trace_end_ts = max(int(event["ts"]) for event in events) if events else 0

    for event in events:
        op = str(event["op"])
        ts = int(event["ts"])
        workflow_id = str(event.get("workflow_id", "workflow"))
        metadata = dict(event.get("metadata") or {})

        if op in {"CREATE", "DERIVE"}:
            state_id = str(event["state_id"])
            state_type = str(event["state_type"])
            module = str(
                metadata.get("context_module")
                or DEFAULT_MODULE_BY_STATE_TYPE.get(state_type, state_type)
            )
            states[state_id] = StateRecord(
                state_id=state_id,
                logical_key=str(event["logical_key"]),
                state_type=state_type,
                workflow_id=workflow_id,
                created_ts=ts,
                size_bytes=int(event["size_bytes"]),
                token_count=int(event["token_count"]),
                version=int(event.get("version", 1)),
                module=module,
                lifecycle_class=str(metadata.get("lifecycle_class", "unknown")),
                is_immutable=bool(metadata.get("is_immutable", False)),
                is_shared=bool(metadata.get("is_shared", False)),
                is_ephemeral=bool(metadata.get("is_ephemeral", False)),
                update_cause=str(metadata.get("update_cause", "unknown")),
                recompute_cost=float(event.get("recompute_cost", 0.0)),
                reload_cost=float(event.get("reload_cost", 0.0)),
                materialization=str(event["materialization"]) if "materialization" in event else None,
                metadata=metadata,
            )
        elif op == "READ":
            state_id = str(event["state_id"])
            if state_id not in states:
                continue
            record = states[state_id]
            record.read_ts.append(ts)
            consumer = str(event["consumer"])
            record.consumers.add(consumer)
            prompt_id = str(metadata.get("prompt_id", f"{consumer}-{ts}"))
            record.read_prompt_ids.append(prompt_id)
            key = (workflow_id, consumer, prompt_id)
            if key not in prompt_calls:
                prompt_calls[key] = PromptCall(
                    workflow_id=workflow_id,
                    consumer=consumer,
                    prompt_id=prompt_id,
                    ts=ts,
                )
            call = prompt_calls[key]
            call.state_ids.append(state_id)
            call.segment_modules.append(
                str(metadata.get("context_module") or record.module)
            )
        elif op == "SUPERSEDE":
            old_state_id = str(event["old_state_id"])
            if old_state_id in states:
                states[old_state_id].superseded_ts = ts
        elif op == "RELEASE":
            state_id = str(event["state_id"])
            if state_id in states and states[state_id].released_ts is None:
                states[state_id].released_ts = ts

    prompt_call_list = sorted(
        prompt_calls.values(),
        key=lambda call: (call.consumer, call.ts, call.prompt_id),
    )
    next_prompt_ts_by_key: Dict[tuple[str, str, str], int | None] = {}
    next_prompt_call_by_key: Dict[tuple[str, str, str], PromptCall | None] = {}
    calls_by_consumer: Dict[str, List[PromptCall]] = defaultdict(list)
    for call in prompt_call_list:
        calls_by_consumer[call.consumer].append(call)
    for consumer_calls in calls_by_consumer.values():
        for index, call in enumerate(consumer_calls):
            next_call = consumer_calls[index + 1] if index + 1 < len(consumer_calls) else None
            next_prompt_ts_by_key[(call.workflow_id, call.consumer, call.prompt_id)] = (
                next_call.ts if next_call is not None else None
            )
            next_prompt_call_by_key[(call.workflow_id, call.consumer, call.prompt_id)] = next_call

    composition = _composition_summary(prompt_call_list, states)
    lifecycle = _lifecycle_summary(states.values(), trace_end_ts)
    lifetime_correlation = _lifetime_correlation_summary(
        prompt_call_list,
        states,
        trace_end_ts,
    )
    mismatch = _mismatch_summary(
        prompt_call_list,
        states,
        next_prompt_ts_by_key,
        trace_end_ts,
    )
    reuse_locality = _reuse_locality_summary(
        prompt_calls=prompt_call_list,
        states=states,
    )
    oracle_abstraction_bridge = _oracle_abstraction_bridge_summary(
        prompt_calls=prompt_call_list,
        states=states,
        next_prompt_ts_by_key=next_prompt_ts_by_key,
        next_prompt_call_by_key=next_prompt_call_by_key,
        trace_end_ts=trace_end_ts,
    )
    return {
        "trace_end_ts": trace_end_ts,
        "state_count": len(states),
        "prompt_call_count": len(prompt_call_list),
        "prompt_composition": composition,
        "lifecycle_characterization": lifecycle,
        "lifetime_correlation": lifetime_correlation,
        "abstraction_mismatch": mismatch,
        "reuse_locality": reuse_locality,
        "oracle_abstraction_bridge": oracle_abstraction_bridge,
        "abstraction_bridge": _legacy_bridge_alias(oracle_abstraction_bridge),
    }


def analyze_runtime_events(events: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not events:
        return _empty_runtime_behavior_summary()

    op_counts: Dict[str, int] = defaultdict(int)
    lookup_status_counts: Dict[str, int] = defaultdict(int)
    role_counts: Dict[str, Dict[str, float | int | str]] = {}
    workflow_counts: Dict[str, Dict[str, float | int | str]] = {}
    execution_context_counts: Dict[str, Dict[str, float | int | str]] = {}
    segment_stats: Dict[str, Dict[str, object]] = {}
    last_timestamp = max(float(event.get("timestamp", 0.0)) for event in events)

    lifecycle_reasons = {
        "release_reclamation",
        "supersede_reclamation",
        "compat_invalidate",
    }

    def ensure_segment(event: Mapping[str, object]) -> Dict[str, object]:
        segment_id = str(event.get("segment_id", "unknown"))
        if segment_id not in segment_stats:
            segment_stats[segment_id] = {
                "segment_id": segment_id,
                "logical_id": str(event.get("logical_id", segment_id)),
                "workflow_id": str(event.get("workflow_id", "workflow")),
                "role": str(event.get("role", "unknown")),
                "module": str(event.get("module", "unknown")),
                "version": int(event.get("version", 0)),
                "size_bytes": int(event.get("size_bytes", 0)),
                "token_count": int(event.get("token_count", 0)),
                "registered_at": None,
                "first_lookup_at": None,
                "last_lookup_at": None,
                "released_at": None,
                "resident_hits": 0,
                "evicted_hits": 0,
                "misses": 0,
                "invalid_lookups": 0,
                "materializations": 0,
                "rematerializations": 0,
                "reuse_count": 0,
                "reused_tokens": 0,
                "materialized_tokens": 0,
                "rematerialized_tokens": 0,
            }
        return segment_stats[segment_id]

    def bump_runtime_bucket(
        buckets: Dict[str, Dict[str, float | int | str]],
        key: str,
        *,
        role: str | None = None,
    ) -> Dict[str, float | int | str]:
        if key not in buckets:
            buckets[key] = {
                "name": key,
                "role": role or key,
                "registrations": 0,
                "resident_hits": 0,
                "evicted_hits": 0,
                "misses": 0,
                "invalid_lookups": 0,
                "materializations": 0,
                "rematerializations": 0,
                "reuse_count": 0,
                "lifecycle_reclaims": 0,
                "policy_reclaims": 0,
                "bytes_reclaimed": 0,
                "reused_tokens": 0,
                "materialized_tokens": 0,
                "rematerialized_tokens": 0,
            }
        return buckets[key]

    for event in events:
        operation = str(event.get("operation", "UNKNOWN"))
        op_counts[operation] += 1
        role = str(event.get("role", "unknown"))
        workflow_id = str(event.get("workflow_id", "workflow"))
        timestamp = float(event.get("timestamp", 0.0))
        size_bytes = int(event.get("size_bytes", 0))
        token_count = int(event.get("token_count", 0))
        reason = str(event.get("reason", "")) if event.get("reason") is not None else ""
        segment = ensure_segment(event)
        role_bucket = bump_runtime_bucket(role_counts, role, role=role)
        workflow_bucket = bump_runtime_bucket(workflow_counts, workflow_id)

        status = event.get("lookup_status")
        if status:
            status_name = str(status)
            lookup_status_counts[status_name] += 1
            if segment["first_lookup_at"] is None:
                segment["first_lookup_at"] = timestamp
            segment["last_lookup_at"] = timestamp
            if status_name == "HIT_RESIDENT":
                role_bucket["resident_hits"] += 1
                workflow_bucket["resident_hits"] += 1
                segment["resident_hits"] = int(segment["resident_hits"]) + 1
            elif status_name == "HIT_EVICTED":
                role_bucket["evicted_hits"] += 1
                workflow_bucket["evicted_hits"] += 1
                segment["evicted_hits"] = int(segment["evicted_hits"]) + 1
            elif status_name == "MISS":
                role_bucket["misses"] += 1
                workflow_bucket["misses"] += 1
                segment["misses"] = int(segment["misses"]) + 1
            elif status_name == "INVALID":
                role_bucket["invalid_lookups"] += 1
                workflow_bucket["invalid_lookups"] += 1
                segment["invalid_lookups"] = int(segment["invalid_lookups"]) + 1

        if operation == "REGISTER":
            role_bucket["registrations"] += 1
            workflow_bucket["registrations"] += 1
            segment["registered_at"] = timestamp
        elif operation == "REUSE":
            role_bucket["reuse_count"] += 1
            workflow_bucket["reuse_count"] += 1
            role_bucket["reused_tokens"] += token_count
            workflow_bucket["reused_tokens"] += token_count
            segment["reuse_count"] = int(segment["reuse_count"]) + 1
            segment["reused_tokens"] = int(segment["reused_tokens"]) + token_count
        elif operation == "MATERIALIZE_INTERNAL":
            role_bucket["materializations"] += 1
            workflow_bucket["materializations"] += 1
            role_bucket["materialized_tokens"] += token_count
            workflow_bucket["materialized_tokens"] += token_count
            segment["materializations"] = int(segment["materializations"]) + 1
            segment["materialized_tokens"] = int(segment["materialized_tokens"]) + token_count
            if reason == "rematerialization_after_eviction":
                role_bucket["rematerializations"] += 1
                workflow_bucket["rematerializations"] += 1
                role_bucket["rematerialized_tokens"] += token_count
                workflow_bucket["rematerialized_tokens"] += token_count
                segment["rematerializations"] = int(segment["rematerializations"]) + 1
                segment["rematerialized_tokens"] = int(segment["rematerialized_tokens"]) + token_count
        elif operation == "RECLAIM":
            if reason == "policy_eviction":
                role_bucket["policy_reclaims"] += 1
                workflow_bucket["policy_reclaims"] += 1
            elif reason in lifecycle_reasons:
                role_bucket["lifecycle_reclaims"] += 1
                workflow_bucket["lifecycle_reclaims"] += 1
            role_bucket["bytes_reclaimed"] += size_bytes
            workflow_bucket["bytes_reclaimed"] += size_bytes
        elif operation == "RELEASE":
            segment["released_at"] = timestamp

        execution_context_digest = event.get("execution_context_digest")
        if execution_context_digest:
            digest = str(execution_context_digest)
            if digest not in execution_context_counts:
                execution_context_counts[digest] = {
                    "execution_context_digest": digest,
                    "workflow_id": workflow_id,
                    "role": role,
                    "segment_id": str(event.get("segment_id", "unknown")),
                    "resident_hits": 0,
                    "evicted_hits": 0,
                    "misses": 0,
                    "materializations": 0,
                    "rematerializations": 0,
                }
            execution_bucket = execution_context_counts[digest]
            if status == "HIT_RESIDENT":
                execution_bucket["resident_hits"] = int(execution_bucket["resident_hits"]) + 1
            elif status == "HIT_EVICTED":
                execution_bucket["evicted_hits"] = int(execution_bucket["evicted_hits"]) + 1
            elif status == "MISS":
                execution_bucket["misses"] = int(execution_bucket["misses"]) + 1
            if operation == "MATERIALIZE_INTERNAL":
                execution_bucket["materializations"] = int(execution_bucket["materializations"]) + 1
                if reason == "rematerialization_after_eviction":
                    execution_bucket["rematerializations"] = int(
                        execution_bucket["rematerializations"]
                    ) + 1

    role_segment_map: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for segment in segment_stats.values():
        role_segment_map[str(segment["role"])].append(segment)

    role_rows = []
    lifetime_rows = []
    for role, bucket in sorted(role_counts.items()):
        segments_for_role = role_segment_map.get(role, [])
        registered_segments = len(segments_for_role)
        registered_but_never_reused = sum(
            int(segment["reuse_count"]) == 0 for segment in segments_for_role
        )
        reused_exactly_once = sum(
            int(segment["reuse_count"]) == 1 for segment in segments_for_role
        )
        reused_more_than_five = sum(
            int(segment["reuse_count"]) > 5 for segment in segments_for_role
        )
        misses = int(bucket["misses"])
        resident_hits = int(bucket["resident_hits"])
        evicted_hits = int(bucket["evicted_hits"])
        invalid_lookups = int(bucket["invalid_lookups"])
        denominator = resident_hits + misses + evicted_hits + invalid_lookups
        role_rows.append(
            {
                "role": role,
                "registrations": int(bucket["registrations"]),
                "resident_hits": resident_hits,
                "evicted_hits": evicted_hits,
                "misses": misses,
                "invalid_lookups": invalid_lookups,
                "materializations": int(bucket["materializations"]),
                "rematerializations": int(bucket["rematerializations"]),
                "reuse_count": int(bucket["reuse_count"]),
                "lifecycle_reclaims": int(bucket["lifecycle_reclaims"]),
                "policy_reclaims": int(bucket["policy_reclaims"]),
                "bytes_reclaimed": int(bucket["bytes_reclaimed"]),
                "reused_tokens": int(bucket["reused_tokens"]),
                "materialized_tokens": int(bucket["materialized_tokens"]),
                "rematerialized_tokens": int(bucket["rematerialized_tokens"]),
                "registered_segments": registered_segments,
                "registered_but_never_reused": registered_but_never_reused,
                "reused_exactly_once": reused_exactly_once,
                "reused_more_than_five": reused_more_than_five,
                "resident_hit_rate": (
                    resident_hits / denominator if denominator else 0.0
                ),
                "token_weighted_reuse_rate": (
                    int(bucket["reused_tokens"])
                    / (int(bucket["reused_tokens"]) + int(bucket["materialized_tokens"]))
                    if (int(bucket["reused_tokens"]) + int(bucket["materialized_tokens"]))
                    else 0.0
                ),
            }
        )

        semantic_lifetimes = []
        registration_to_first_lookup = []
        lookup_spans = []
        for segment in segments_for_role:
            registered_at = segment["registered_at"]
            first_lookup_at = segment["first_lookup_at"]
            last_lookup_at = segment["last_lookup_at"]
            released_at = segment["released_at"]
            if registered_at is not None:
                semantic_lifetimes.append(
                    float(released_at if released_at is not None else last_timestamp)
                    - float(registered_at)
                )
            if registered_at is not None and first_lookup_at is not None:
                registration_to_first_lookup.append(
                    float(first_lookup_at) - float(registered_at)
                )
            if first_lookup_at is not None and last_lookup_at is not None:
                lookup_spans.append(float(last_lookup_at) - float(first_lookup_at))
        lifetime_rows.append(
            {
                "role": role,
                "count": registered_segments,
                "avg_semantic_lifetime": (
                    statistics.fmean(semantic_lifetimes) if semantic_lifetimes else 0.0
                ),
                "avg_registration_to_first_lookup": (
                    statistics.fmean(registration_to_first_lookup)
                    if registration_to_first_lookup
                    else 0.0
                ),
                "avg_lookup_span": (
                    statistics.fmean(lookup_spans) if lookup_spans else 0.0
                ),
                "registered_but_never_reused_ratio": (
                    registered_but_never_reused / registered_segments
                    if registered_segments
                    else 0.0
                ),
            }
        )

    workflow_rows = []
    for workflow_id, bucket in sorted(workflow_counts.items()):
        misses = int(bucket["misses"])
        resident_hits = int(bucket["resident_hits"])
        evicted_hits = int(bucket["evicted_hits"])
        invalid_lookups = int(bucket["invalid_lookups"])
        denominator = resident_hits + misses + evicted_hits + invalid_lookups
        workflow_rows.append(
            {
                "workflow_id": workflow_id,
                "registrations": int(bucket["registrations"]),
                "resident_hits": resident_hits,
                "evicted_hits": evicted_hits,
                "misses": misses,
                "invalid_lookups": invalid_lookups,
                "materializations": int(bucket["materializations"]),
                "rematerializations": int(bucket["rematerializations"]),
                "reuse_count": int(bucket["reuse_count"]),
                "lifecycle_reclaims": int(bucket["lifecycle_reclaims"]),
                "policy_reclaims": int(bucket["policy_reclaims"]),
                "bytes_reclaimed": int(bucket["bytes_reclaimed"]),
                "reused_tokens": int(bucket["reused_tokens"]),
                "materialized_tokens": int(bucket["materialized_tokens"]),
                "rematerialized_tokens": int(bucket["rematerialized_tokens"]),
                "resident_hit_rate": (
                    resident_hits / denominator if denominator else 0.0
                ),
                "token_weighted_reuse_rate": (
                    int(bucket["reused_tokens"])
                    / (int(bucket["reused_tokens"]) + int(bucket["materialized_tokens"]))
                    if (int(bucket["reused_tokens"]) + int(bucket["materialized_tokens"]))
                    else 0.0
                ),
            }
        )

    segment_rows = []
    for segment in sorted(
        segment_stats.values(),
        key=lambda item: (
            str(item["workflow_id"]),
            str(item["role"]),
            str(item["segment_id"]),
        ),
    ):
        registered_at = segment["registered_at"]
        released_at = segment["released_at"]
        first_lookup_at = segment["first_lookup_at"]
        last_lookup_at = segment["last_lookup_at"]
        segment_rows.append(
            {
                **segment,
                "semantic_lifetime": (
                    float(released_at if released_at is not None else last_timestamp)
                    - float(registered_at)
                    if registered_at is not None
                    else 0.0
                ),
                "registration_to_first_lookup": (
                    float(first_lookup_at) - float(registered_at)
                    if registered_at is not None and first_lookup_at is not None
                    else None
                ),
                "lookup_span": (
                    float(last_lookup_at) - float(first_lookup_at)
                    if first_lookup_at is not None and last_lookup_at is not None
                    else None
                ),
                "reused_tokens": int(segment["reused_tokens"]),
                "materialized_tokens": int(segment["materialized_tokens"]),
            }
        )

    resident_hits = lookup_status_counts.get("HIT_RESIDENT", 0)
    evicted_hits = lookup_status_counts.get("HIT_EVICTED", 0)
    misses = lookup_status_counts.get("MISS", 0)
    invalid_lookups = lookup_status_counts.get("INVALID", 0)
    resident_hit_denominator = resident_hits + evicted_hits + misses + invalid_lookups
    registered_segments_total = len(segment_rows)
    registered_but_never_reused = sum(
        int(segment["reuse_count"]) == 0 for segment in segment_rows
    )
    reused_exactly_once = sum(
        int(segment["reuse_count"]) == 1 for segment in segment_rows
    )
    reused_more_than_five = sum(
        int(segment["reuse_count"]) > 5 for segment in segment_rows
    )
    reused_tokens = sum(int(segment["reused_tokens"]) for segment in segment_rows)
    materialized_tokens = sum(int(segment["materialized_tokens"]) for segment in segment_rows)
    rematerialized_tokens = sum(int(segment["rematerialized_tokens"]) for segment in segment_rows)

    return {
        "event_count": len(events),
        "op_counts": dict(sorted(op_counts.items())),
        "lookup_status_counts": dict(sorted(lookup_status_counts.items())),
        "resident_hit_rate": (
            resident_hits / resident_hit_denominator if resident_hit_denominator else 0.0
        ),
        "registrations": op_counts.get("REGISTER", 0),
        "resident_hits": resident_hits,
        "evicted_hits": evicted_hits,
        "misses": misses,
        "invalid_lookups": invalid_lookups,
        "materializations": op_counts.get("MATERIALIZE_INTERNAL", 0),
        "rematerializations": sum(
            int(segment["rematerializations"]) for segment in segment_rows
        ),
        "reuse_count": op_counts.get("REUSE", 0),
        "reused_tokens": reused_tokens,
        "materialized_tokens": materialized_tokens,
        "rematerialized_tokens": rematerialized_tokens,
        "token_weighted_reuse_rate": (
            reused_tokens / (reused_tokens + materialized_tokens)
            if (reused_tokens + materialized_tokens)
            else 0.0
        ),
        "reuse_benefit_ratio": (
            reused_tokens / (reused_tokens + materialized_tokens)
            if (reused_tokens + materialized_tokens)
            else 0.0
        ),
        "prompt_token_reuse_fraction": 0.0,
        "lifecycle_reclaims": sum(
            int(bucket["lifecycle_reclaims"]) for bucket in role_counts.values()
        ),
        "policy_reclaims": sum(
            int(bucket["policy_reclaims"]) for bucket in role_counts.values()
        ),
        "bytes_reclaimed": sum(
            int(bucket["bytes_reclaimed"]) for bucket in role_counts.values()
        ),
        "registered_segments": registered_segments_total,
        "segment_reuse_buckets": {
            "registered_segments": registered_segments_total,
            "registered_but_never_reused": registered_but_never_reused,
            "reused_exactly_once": reused_exactly_once,
            "reused_more_than_five": reused_more_than_five,
        },
        "role_rows": role_rows,
        "workflow_rows": workflow_rows,
        "lifetime_rows": lifetime_rows,
        "segment_rows": segment_rows,
        "execution_context_rows": sorted(
            execution_context_counts.values(),
            key=lambda item: (
                str(item["workflow_id"]),
                str(item["segment_id"]),
                str(item["execution_context_digest"]),
            ),
        ),
    }


def analyze_backend_call_records(
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not records:
        return _empty_backend_latency_summary()

    total_duration_ms = sum(float(record.get("duration_ms", 0.0)) for record in records)
    total_backend_roundtrip_ms = sum(
        float(record.get("backend_roundtrip_ms", 0.0)) for record in records
    )
    total_frontend_message_build_ms = sum(
        float(record.get("frontend_message_build_ms", 0.0)) for record in records
    )
    total_frontend_token_estimate_ms = sum(
        float(record.get("frontend_token_estimate_ms", 0.0)) for record in records
    )
    total_frontend_overhead_ms = sum(
        float(record.get("frontend_overhead_ms", 0.0)) for record in records
    )
    total_prompt_tokens = sum(int(record.get("prompt_tokens") or 0) for record in records)
    total_completion_tokens = sum(
        int(record.get("completion_tokens") or 0) for record in records
    )
    total_tokens = sum(int(record.get("total_tokens") or 0) for record in records)
    frontend_cache_hits = sum(
        1 for record in records if bool(record.get("frontend_cache_hit", False))
    )
    total_system_prompt_tokens_estimate = sum(
        int(record.get("system_prompt_tokens_estimate") or 0) for record in records
    )
    total_user_prompt_tokens_estimate = sum(
        int(record.get("user_prompt_tokens_estimate") or 0) for record in records
    )
    total_ordered_segment_tokens = sum(
        int(record.get("ordered_segment_tokens") or 0) for record in records
    )
    total_request_segment_tokens = sum(
        int(record.get("request_segment_tokens") or 0) for record in records
    )
    total_assembled_prompt_tokens_estimate = sum(
        int(record.get("assembled_prompt_tokens_estimate") or 0) for record in records
    )
    total_prompt_payload_tokens_estimate = sum(
        int(record.get("prompt_payload_tokens_estimate") or 0) for record in records
    )
    total_duplicate_prompt_tokens_estimate = sum(
        int(record.get("duplicate_prompt_tokens_estimate") or 0) for record in records
    )
    grouped: Dict[tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record.get("step_name", "unknown")),
                str(record.get("prompt_mode", "unknown")),
            )
        ].append(record)

    step_rows = []
    for (step_name, prompt_mode), step_records in sorted(grouped.items()):
        step_duration_ms = sum(
            float(record.get("duration_ms", 0.0)) for record in step_records
        )
        step_backend_roundtrip_ms = sum(
            float(record.get("backend_roundtrip_ms", 0.0)) for record in step_records
        )
        step_frontend_message_build_ms = sum(
            float(record.get("frontend_message_build_ms", 0.0))
            for record in step_records
        )
        step_frontend_token_estimate_ms = sum(
            float(record.get("frontend_token_estimate_ms", 0.0))
            for record in step_records
        )
        step_frontend_overhead_ms = sum(
            float(record.get("frontend_overhead_ms", 0.0)) for record in step_records
        )
        step_prompt_tokens = sum(
            int(record.get("prompt_tokens") or 0) for record in step_records
        )
        step_completion_tokens = sum(
            int(record.get("completion_tokens") or 0) for record in step_records
        )
        step_total_tokens = sum(
            int(record.get("total_tokens") or 0) for record in step_records
        )
        step_frontend_cache_hits = sum(
            1 for record in step_records if bool(record.get("frontend_cache_hit", False))
        )
        step_system_prompt_tokens_estimate = sum(
            int(record.get("system_prompt_tokens_estimate") or 0) for record in step_records
        )
        step_user_prompt_tokens_estimate = sum(
            int(record.get("user_prompt_tokens_estimate") or 0) for record in step_records
        )
        step_ordered_segment_tokens = sum(
            int(record.get("ordered_segment_tokens") or 0) for record in step_records
        )
        step_request_segment_tokens = sum(
            int(record.get("request_segment_tokens") or 0) for record in step_records
        )
        step_assembled_prompt_tokens_estimate = sum(
            int(record.get("assembled_prompt_tokens_estimate") or 0) for record in step_records
        )
        step_prompt_payload_tokens_estimate = sum(
            int(record.get("prompt_payload_tokens_estimate") or 0) for record in step_records
        )
        step_duplicate_prompt_tokens_estimate = sum(
            int(record.get("duplicate_prompt_tokens_estimate") or 0) for record in step_records
        )
        step_rows.append(
            {
                "step_name": step_name,
                "prompt_mode": prompt_mode,
                "request_count": len(step_records),
                "total_duration_ms": step_duration_ms,
                "avg_duration_ms": step_duration_ms / len(step_records),
                "total_backend_roundtrip_ms": step_backend_roundtrip_ms,
                "avg_backend_roundtrip_ms": step_backend_roundtrip_ms / len(step_records),
                "total_frontend_message_build_ms": step_frontend_message_build_ms,
                "avg_frontend_message_build_ms": step_frontend_message_build_ms
                / len(step_records),
                "total_frontend_token_estimate_ms": step_frontend_token_estimate_ms,
                "avg_frontend_token_estimate_ms": step_frontend_token_estimate_ms
                / len(step_records),
                "total_frontend_overhead_ms": step_frontend_overhead_ms,
                "avg_frontend_overhead_ms": step_frontend_overhead_ms
                / len(step_records),
                "total_prompt_tokens": step_prompt_tokens,
                "total_completion_tokens": step_completion_tokens,
                "total_tokens": step_total_tokens,
                "avg_prompt_tokens": (
                    step_prompt_tokens / len(step_records) if step_records else 0.0
                ),
                "duration_ms_per_1k_prompt_tokens": (
                    step_duration_ms / (step_prompt_tokens / 1000.0)
                    if step_prompt_tokens
                    else 0.0
                ),
                "frontend_cache_hits": step_frontend_cache_hits,
                "frontend_cache_hit_rate": (
                    step_frontend_cache_hits / len(step_records) if step_records else 0.0
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
        "request_count": len(records),
        "total_duration_ms": total_duration_ms,
        "avg_duration_ms": total_duration_ms / len(records),
        "total_backend_roundtrip_ms": total_backend_roundtrip_ms,
        "avg_backend_roundtrip_ms": total_backend_roundtrip_ms / len(records),
        "total_frontend_message_build_ms": total_frontend_message_build_ms,
        "avg_frontend_message_build_ms": total_frontend_message_build_ms / len(records),
        "total_frontend_token_estimate_ms": total_frontend_token_estimate_ms,
        "avg_frontend_token_estimate_ms": total_frontend_token_estimate_ms / len(records),
        "total_frontend_overhead_ms": total_frontend_overhead_ms,
        "avg_frontend_overhead_ms": total_frontend_overhead_ms / len(records),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": total_prompt_tokens / len(records),
        "duration_ms_per_1k_prompt_tokens": (
            total_duration_ms / (total_prompt_tokens / 1000.0)
            if total_prompt_tokens
            else 0.0
        ),
        "frontend_cache_hits": frontend_cache_hits,
        "frontend_cache_hit_rate": frontend_cache_hits / len(records),
        "total_system_prompt_tokens_estimate": total_system_prompt_tokens_estimate,
        "total_user_prompt_tokens_estimate": total_user_prompt_tokens_estimate,
        "total_ordered_segment_tokens": total_ordered_segment_tokens,
        "total_request_segment_tokens": total_request_segment_tokens,
        "total_assembled_prompt_tokens_estimate": total_assembled_prompt_tokens_estimate,
        "total_prompt_payload_tokens_estimate": total_prompt_payload_tokens_estimate,
        "total_duplicate_prompt_tokens_estimate": total_duplicate_prompt_tokens_estimate,
        "step_rows": step_rows,
    }


def analyze_trace_paths(
    trace_paths: Sequence[str | Path],
    *,
    trace_metadata_by_path: Mapping[str, Mapping[str, object]] | None = None,
    runtime_event_paths_by_trace_path: Mapping[str, str | Path] | None = None,
    backend_call_paths_by_trace_path: Mapping[str, str | Path] | None = None,
    run_summary: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    per_trace = {}
    analyses = []
    analyses_by_provider: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for path in trace_paths:
        resolved = str(Path(path).resolve())
        analysis = analyze_trace_events(load_trace_events(path))
        if runtime_event_paths_by_trace_path is not None:
            runtime_event_path = runtime_event_paths_by_trace_path.get(resolved)
            if runtime_event_path:
                analysis["runtime_behavior"] = analyze_runtime_events(
                    load_runtime_events(runtime_event_path)
                )
        if backend_call_paths_by_trace_path is not None:
            backend_call_path = backend_call_paths_by_trace_path.get(resolved)
            if backend_call_path:
                analysis["backend_latency"] = analyze_backend_call_records(
                    load_backend_call_records(backend_call_path)
                )
        per_trace[resolved] = analysis
        analyses.append(analysis)
        if trace_metadata_by_path is not None and resolved in trace_metadata_by_path:
            provider = str(trace_metadata_by_path[resolved].get("provider", "unknown"))
            analyses_by_provider[provider].append(analysis)

    report: Dict[str, object] = {
        "trace_count": len(analyses),
        "per_trace": per_trace,
        "aggregate": _aggregate_analyses(analyses),
    }
    if analyses_by_provider:
        report["providers"] = {
            provider: {
                "trace_count": len(provider_analyses),
                "aggregate": _aggregate_analyses(provider_analyses),
            }
            for provider, provider_analyses in sorted(analyses_by_provider.items())
        }
    if run_summary is not None:
        report["run_summary"] = dict(run_summary)
    return report


def render_markdown_report(report: Mapping[str, object]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Trace Analysis",
        "",
        f"- Traces: {report['trace_count']}",
        f"- Prompt calls: {aggregate['prompt_call_count']}",
        f"- States: {aggregate['state_count']}",
        "",
    ]
    if "run_summary" in report:
        run_summary = report["run_summary"]
        lines.extend(
            [
                "## Run Summary",
                "",
                f"- Records: {run_summary['record_count']}",
                f"- Valid traces: {run_summary['valid_trace_count']}",
                f"- Real-backend traces: {run_summary['real_backend_trace_count']}",
                f"- Stub traces: {run_summary['stub_trace_count']}",
                f"- Missing trace paths: {run_summary['missing_trace_path_count']}",
                f"- Runtime event logs: {run_summary.get('runtime_event_path_count', 0)}",
                f"- Backend call logs: {run_summary.get('backend_call_path_count', 0)}",
                "",
            ]
        )
        provider_counts = run_summary.get("provider_counts", {})
        if provider_counts:
            lines.extend(
                [
                    "### Providers",
                    "",
                    "| Provider | Count |",
                    "| - | -: |",
                ]
            )
            for provider, count in provider_counts.items():
                lines.append(f"| {provider} | {count} |")
            lines.append("")

    lines.extend(
        [
            "## Prompt Composition",
            "",
            "| Module | Prompt Presence Rate | Avg Segments / Prompt | Avg Bytes / Prompt | Immutable | Shared | Ephemeral |",
            "| - | -: | -: | -: | -: | -: | -: |",
        ]
    )
    for row in aggregate["prompt_composition"]["module_rows"]:
        lines.append(
            f"| {row['module']} | {row['prompt_presence_rate']:.2f} | {row['avg_segments_per_prompt']:.2f} | "
            f"{row['avg_bytes_per_prompt']:.1f} | {row['immutable_fraction']:.2f} | "
            f"{row['shared_fraction']:.2f} | {row['ephemeral_fraction']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Lifecycle Characterization",
            "",
            "| Module | Count | Avg Lifetime | Median Lifetime | Avg Reads | Superseded | Released |",
            "| - | -: | -: | -: | -: | -: | -: |",
        ]
    )
    for row in aggregate["lifecycle_characterization"]["module_rows"]:
        lines.append(
            f"| {row['module']} | {row['count']} | {row['avg_lifetime']:.2f} | {row['median_lifetime']:.2f} | "
            f"{row['avg_reads']:.2f} | {row['superseded_fraction']:.2f} | {row['released_fraction']:.2f} |"
        )

    lifetime_correlation = aggregate["lifetime_correlation"]
    lines.extend(
        [
            "",
            "## Lifetime Correlation Matrix",
            "",
            f"- Prompt vectors: {lifetime_correlation['prompt_vector_count']}",
            f"- Modules: {', '.join(lifetime_correlation['module_order']) if lifetime_correlation['module_order'] else 'none'}",
            "",
            "### Correlation",
            "",
        ]
    )
    lines.extend(
        _render_matrix_markdown(
            lifetime_correlation["module_order"],
            lifetime_correlation["correlation_matrix"],
            lambda value: _format_matrix_value(value, precision=2),
        )
    )
    lines.extend(
        [
            "",
            "### Covariance",
            "",
        ]
    )
    lines.extend(
        _render_matrix_markdown(
            lifetime_correlation["module_order"],
            lifetime_correlation["covariance_matrix"],
            lambda value: _format_matrix_value(value, precision=2),
        )
    )
    lines.extend(
        [
            "",
            "### Heatmap",
            "",
        ]
    )
    lines.extend(
        _render_matrix_markdown(
            lifetime_correlation["module_order"],
            lifetime_correlation["heatmap_matrix"],
            lambda value: str(value),
        )
    )

    mismatch = aggregate["abstraction_mismatch"]
    lines.extend(
        [
            "",
            "## Abstraction Mismatch",
            "",
            f"- Mixed lifecycle prompt rate: {mismatch['mixed_lifecycle_prompt_rate']:.2f}",
            f"- Monolithic invalidation events: {mismatch['monolithic_invalidation_events']}",
            f"- Stale bytes before next prompt: {mismatch['stale_bytes_before_next_prompt']}",
            f"- Reusable live bytes co-resident with stale bytes: {mismatch['reusable_live_bytes']}",
            f"- Pinned live fraction: {mismatch['pinned_live_fraction']:.2f}",
            f"- Fragmentation loss: {mismatch['fragmentation_loss']:.2f}",
            f"- Avg lifetime spread per prompt: {mismatch['avg_lifetime_spread']:.2f}",
        ]
    )

    reuse_locality = aggregate.get("reuse_locality")
    if reuse_locality is not None:
        lines.extend(
            [
                "",
                "## Reuse Locality",
                "",
                f"- Segments: {reuse_locality['segment_count']}",
                f"- Avg accesses / segment: {reuse_locality['avg_accesses_per_segment']:.2f}",
                f"- Avg revisits / segment: {reuse_locality['avg_revisits_per_segment']:.2f}",
                f"- Revisited segment fraction: {reuse_locality['revisited_segment_fraction']:.2f}",
                f"- Avg prompt revisit distance: {reuse_locality['avg_prompt_revisit_distance']:.2f}",
                "",
                "### By Module",
                "",
                "| Module | Count | Avg Accesses / Segment | Avg Revisits / Segment | Revisited Fraction | Avg Prompt Revisit Distance |",
                "| - | -: | -: | -: | -: | -: |",
            ]
        )
        for row in reuse_locality["module_rows"]:
            lines.append(
                f"| {row['module']} | {row['count']} | {row['avg_accesses_per_segment']:.2f} | "
                f"{row['avg_revisits_per_segment']:.2f} | {row['revisited_segment_fraction']:.2f} | "
                f"{row['avg_prompt_revisit_distance']:.2f} |"
            )

    bridge = aggregate.get("oracle_abstraction_bridge", aggregate["abstraction_bridge"])
    lines.extend(
        [
            "",
            "## Oracle Abstraction Bridge",
            "",
            "- Interpretation: offline oracle upper bound derived from future-complete traces.",
            f"- Monolithic peak HBM bytes: {bridge['monolithic_peak_hbm_bytes']}",
            f"- Ideal segment-aware peak HBM bytes: {bridge['ideal_segment_peak_hbm_bytes']}",
            f"- Monolithic pinned live bytes: {bridge['monolithic_pinned_live_bytes']}",
            f"- Ideal segment-aware pinned live bytes: {bridge['ideal_segment_pinned_live_bytes']}",
            f"- Monolithic stale retained bytes: {bridge['monolithic_stale_retained_bytes']}",
            f"- Ideal segment-aware stale retained bytes: {bridge['ideal_segment_stale_retained_bytes']}",
            f"- Monolithic rematerialized live bytes: {bridge['monolithic_rematerialized_live_bytes']}",
            f"- Ideal segment-aware rematerialized live bytes: {bridge['ideal_segment_rematerialized_live_bytes']}",
            f"- Monolithic reload bytes: {bridge['monolithic_reload_bytes']}",
            f"- Ideal segment-aware reload bytes: {bridge['ideal_segment_reload_bytes']}",
            f"- Monolithic recompute bytes: {bridge['monolithic_recompute_bytes']}",
            f"- Ideal segment-aware recompute bytes: {bridge['ideal_segment_recompute_bytes']}",
            f"- Monolithic service cost units: {bridge['monolithic_service_cost_units']:.2f}",
            f"- Ideal segment-aware service cost units: {bridge['ideal_segment_service_cost_units']:.2f}",
            f"- Oracle peak-HBM savings fraction: {bridge['oracle_peak_hbm_savings_fraction']:.2f}",
            f"- Oracle pinned-live savings fraction: {bridge['oracle_pinned_live_savings_fraction']:.2f}",
            f"- Oracle reload savings fraction: {bridge['oracle_reload_savings_fraction']:.2f}",
            f"- Oracle recompute savings fraction: {bridge['oracle_recompute_savings_fraction']:.2f}",
            f"- Oracle rematerialization savings fraction: {bridge['oracle_rematerialization_savings_fraction']:.2f}",
            f"- Oracle service-cost savings fraction: {bridge['oracle_service_cost_savings_fraction']:.2f}",
        ]
    )

    runtime_behavior = aggregate.get("runtime_behavior")
    if runtime_behavior is not None:
        reuse_benefit_ratio = float(
            runtime_behavior.get(
                "reuse_benefit_ratio",
                runtime_behavior.get("token_weighted_reuse_rate", 0.0),
            )
        )
        prompt_token_reuse_fraction = float(
            runtime_behavior.get("prompt_token_reuse_fraction", 0.0)
        )
        lines.extend(
            [
                "",
                "## Practical Runtime",
                "",
                f"- Runtime events: {runtime_behavior['event_count']}",
                f"- Resident hit rate: {runtime_behavior['resident_hit_rate']:.2f}",
                f"- Resident hits: {runtime_behavior['resident_hits']}",
                f"- Evicted hits: {runtime_behavior['evicted_hits']}",
                f"- Misses: {runtime_behavior['misses']}",
                f"- Invalid lookups: {runtime_behavior['invalid_lookups']}",
                f"- Materializations: {runtime_behavior['materializations']}",
                f"- Rematerializations: {runtime_behavior['rematerializations']}",
                f"- Reuse count: {runtime_behavior['reuse_count']}",
                f"- Reused tokens: {runtime_behavior['reused_tokens']}",
                f"- Materialized tokens: {runtime_behavior['materialized_tokens']}",
                f"- Token-weighted reuse rate: {runtime_behavior['token_weighted_reuse_rate']:.2f}",
                f"- Reuse benefit ratio: {reuse_benefit_ratio:.2f}",
                f"- Prompt-token reuse fraction: {prompt_token_reuse_fraction:.2f}",
                f"- Lifecycle reclaims: {runtime_behavior['lifecycle_reclaims']}",
                f"- Policy reclaims: {runtime_behavior['policy_reclaims']}",
                f"- Bytes reclaimed: {runtime_behavior['bytes_reclaimed']}",
                "",
                "### Reuse By Role",
                "",
                "| Role | Registrations | Resident Hits | Misses | Materializations | Reused Tokens | Materialized Tokens | Token-Weighted Reuse | Rematerializations | Lifecycle Reclaims | Policy Reclaims | Never Reused | Reused Once | Reused >5 | Resident Hit Rate |",
                "| - | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |",
            ]
        )
        for row in runtime_behavior["role_rows"]:
            lines.append(
                f"| {row['role']} | {row['registrations']} | {row['resident_hits']} | {row['misses']} | "
                f"{row['materializations']} | {row['reused_tokens']} | {row['materialized_tokens']} | "
                f"{row['token_weighted_reuse_rate']:.2f} | {row['rematerializations']} | {row['lifecycle_reclaims']} | "
                f"{row['policy_reclaims']} | {row['registered_but_never_reused']} | {row['reused_exactly_once']} | "
                f"{row['reused_more_than_five']} | {row['resident_hit_rate']:.2f} |"
            )
        lines.extend(
            [
                "",
                "### Segment Lifetime Vs Reuse",
                "",
                "| Role | Count | Avg Semantic Lifetime | Avg Registration->First Lookup | Avg Lookup Span | Never Reused Ratio |",
                "| - | -: | -: | -: | -: | -: |",
            ]
        )
        for row in runtime_behavior["lifetime_rows"]:
            lines.append(
                f"| {row['role']} | {row['count']} | {row['avg_semantic_lifetime']:.2f} | "
                f"{row['avg_registration_to_first_lookup']:.2f} | {row['avg_lookup_span']:.2f} | "
                f"{row['registered_but_never_reused_ratio']:.2f} |"
            )

    backend_latency = aggregate.get("backend_latency")
    if backend_latency is not None:
        lines.extend(
            [
                "",
                "## Request Latency",
                "",
                f"- Requests: {backend_latency['request_count']}",
                f"- Total duration ms: {backend_latency['total_duration_ms']:.2f}",
                f"- Average duration ms: {backend_latency['avg_duration_ms']:.2f}",
                f"- Backend round-trip ms: {backend_latency['total_backend_roundtrip_ms']:.2f}",
                f"- Frontend message-build ms: {backend_latency['total_frontend_message_build_ms']:.2f}",
                f"- Frontend token-estimate ms: {backend_latency['total_frontend_token_estimate_ms']:.2f}",
                f"- Frontend overhead ms: {backend_latency['total_frontend_overhead_ms']:.2f}",
                f"- Total prompt tokens: {backend_latency['total_prompt_tokens']}",
                f"- Total completion tokens: {backend_latency['total_completion_tokens']}",
                f"- Avg prompt tokens / request: {backend_latency['avg_prompt_tokens']:.2f}",
                f"- Duration ms per 1k prompt tokens: {backend_latency['duration_ms_per_1k_prompt_tokens']:.2f}",
                f"- Frontend cache hit rate: {backend_latency['frontend_cache_hit_rate']:.2f}",
                f"- System prompt tokens (est.): {backend_latency['total_system_prompt_tokens_estimate']}",
                f"- User instruction tokens (est.): {backend_latency['total_user_prompt_tokens_estimate']}",
                f"- Ordered segment tokens: {backend_latency['total_ordered_segment_tokens']}",
                f"- Request segment tokens: {backend_latency['total_request_segment_tokens']}",
                f"- Assembled prompt tokens (est.): {backend_latency['total_assembled_prompt_tokens_estimate']}",
                f"- Prompt payload tokens (est.): {backend_latency['total_prompt_payload_tokens_estimate']}",
                f"- Duplicate prompt tokens (est.): {backend_latency['total_duplicate_prompt_tokens_estimate']}",
                "",
                "### By Step",
                "",
                "| Step | Prompt Mode | Requests | Avg Duration ms | Frontend Overhead ms | Backend Round-trip ms | Total Prompt Tokens | User Tokens | Request Segment Tokens | Duplicate Tokens (est.) | Assembled Tokens (est.) | ms / 1k Prompt Tokens | Frontend Cache Hit Rate |",
                "| - | - | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |",
            ]
        )
        for row in backend_latency["step_rows"]:
            lines.append(
                f"| {row['step_name']} | {row['prompt_mode']} | {row['request_count']} | "
                f"{row['avg_duration_ms']:.2f} | {row['avg_frontend_overhead_ms']:.2f} | "
                f"{row['avg_backend_roundtrip_ms']:.2f} | {row['total_prompt_tokens']} | "
                f"{row['user_prompt_tokens_estimate']} | {row['request_segment_tokens']} | "
                f"{row['duplicate_prompt_tokens_estimate']} | {row['assembled_prompt_tokens_estimate']} | "
                f"{row['duration_ms_per_1k_prompt_tokens']:.2f} | "
                f"{row['frontend_cache_hit_rate']:.2f} |"
            )

    if "providers" in report:
        lines.extend(
            [
                "",
                "## Provider Breakdown",
                "",
                "| Provider | Traces | Mixed Lifecycle Rate | Pinned Live Fraction | Fragmentation Loss |",
                "| - | -: | -: | -: | -: |",
            ]
        )
        for provider, provider_report in report["providers"].items():
            provider_aggregate = provider_report["aggregate"]
            provider_mismatch = provider_aggregate["abstraction_mismatch"]
            lines.append(
                f"| {provider} | {provider_report['trace_count']} | "
                f"{provider_mismatch['mixed_lifecycle_prompt_rate']:.2f} | "
                f"{provider_mismatch['pinned_live_fraction']:.2f} | "
                f"{provider_mismatch['fragmentation_loss']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def _composition_summary(
    prompt_calls: Sequence[PromptCall],
    states: Mapping[str, StateRecord],
) -> Dict[str, object]:
    total_prompts = len(prompt_calls) or 1
    module_counts: Dict[str, int] = defaultdict(int)
    module_segments: Dict[str, int] = defaultdict(int)
    module_bytes: Dict[str, int] = defaultdict(int)
    module_immutable: Dict[str, int] = defaultdict(int)
    module_shared: Dict[str, int] = defaultdict(int)
    module_ephemeral: Dict[str, int] = defaultdict(int)

    for call in prompt_calls:
        seen_modules = set()
        for state_id in call.state_ids:
            record = states[state_id]
            module = record.module
            module_segments[module] += 1
            module_bytes[module] += record.size_bytes
            module_immutable[module] += int(record.is_immutable)
            module_shared[module] += int(record.is_shared)
            module_ephemeral[module] += int(record.is_ephemeral)
            seen_modules.add(module)
        for module in seen_modules:
            module_counts[module] += 1

    module_rows = []
    for module in sorted(module_segments):
        segment_count = module_segments[module]
        module_rows.append(
            {
                "module": module,
                "prompt_presence_rate": module_counts[module] / total_prompts,
                "avg_segments_per_prompt": segment_count / total_prompts,
                "avg_bytes_per_prompt": module_bytes[module] / total_prompts,
                "immutable_fraction": module_immutable[module] / segment_count,
                "shared_fraction": module_shared[module] / segment_count,
                "ephemeral_fraction": module_ephemeral[module] / segment_count,
            }
        )
    return {"module_rows": module_rows}


def _lifecycle_summary(
    state_records: Iterable[StateRecord],
    trace_end_ts: int,
) -> Dict[str, object]:
    by_module: Dict[str, List[StateRecord]] = defaultdict(list)
    for record in state_records:
        by_module[record.module].append(record)

    module_rows = []
    for module, records in sorted(by_module.items()):
        lifetimes = [record.lifetime(trace_end_ts) for record in records]
        reads = [len(record.read_ts) for record in records]
        module_rows.append(
            {
                "module": module,
                "count": len(records),
                "avg_lifetime": statistics.fmean(lifetimes) if lifetimes else 0.0,
                "median_lifetime": statistics.median(lifetimes) if lifetimes else 0.0,
                "avg_reads": statistics.fmean(reads) if reads else 0.0,
                "superseded_fraction": sum(
                    record.superseded_ts is not None for record in records
                )
                / len(records),
                "released_fraction": sum(
                    record.released_ts is not None for record in records
                )
                / len(records),
            }
        )
    return {"module_rows": module_rows}


def _lifetime_correlation_summary(
    prompt_calls: Sequence[PromptCall],
    states: Mapping[str, StateRecord],
    trace_end_ts: int,
) -> Dict[str, object]:
    prompt_vectors = []
    for call in prompt_calls:
        module_values: Dict[str, List[int]] = defaultdict(list)
        for state_id in call.state_ids:
            record = states[state_id]
            module_values[record.module].append(max(0, record.end_ts(trace_end_ts) - call.ts))
        if module_values:
            prompt_vectors.append(
                {
                    module: statistics.fmean(values)
                    for module, values in sorted(module_values.items())
                    if values
                }
            )
    return _lifetime_correlation_summary_from_vectors(prompt_vectors)


def _mismatch_summary(
    prompt_calls: Sequence[PromptCall],
    states: Mapping[str, StateRecord],
    next_prompt_ts_by_key: Mapping[tuple[str, str, str], int | None],
    trace_end_ts: int,
) -> Dict[str, object]:
    mixed_prompts = 0
    evaluable_prompts = 0
    stale_bytes_before_next_prompt = 0
    reusable_live_bytes = 0
    total_live_bytes_in_evaluable_prompts = 0
    lifetime_spreads = []

    for call in prompt_calls:
        next_prompt_ts = next_prompt_ts_by_key.get(
            (call.workflow_id, call.consumer, call.prompt_id)
        )
        remaining_lifetimes = []
        if next_prompt_ts is None:
            for state_id in call.state_ids:
                record = states[state_id]
                remaining_lifetimes.append(max(0, record.end_ts(trace_end_ts) - call.ts))
            if remaining_lifetimes:
                lifetime_spreads.append(max(remaining_lifetimes) - min(remaining_lifetimes))
            continue

        evaluable_prompts += 1
        stale_states = []
        live_states = []
        for state_id in call.state_ids:
            record = states[state_id]
            end_ts = record.end_ts(trace_end_ts)
            remaining_lifetimes.append(max(0, end_ts - call.ts))
            if end_ts < next_prompt_ts:
                stale_states.append(record)
            else:
                live_states.append(record)
        if remaining_lifetimes:
            lifetime_spreads.append(max(remaining_lifetimes) - min(remaining_lifetimes))
        total_live_bytes_in_evaluable_prompts += sum(
            record.size_bytes for record in live_states
        )
        if stale_states and live_states:
            mixed_prompts += 1
            stale_bytes_before_next_prompt += sum(
                record.size_bytes for record in stale_states
            )
            reusable_live_bytes += sum(record.size_bytes for record in live_states)

    fragmentation_denominator = reusable_live_bytes + stale_bytes_before_next_prompt
    return {
        "evaluable_prompt_count": evaluable_prompts,
        "monolithic_invalidation_events": mixed_prompts,
        "mixed_lifecycle_prompt_rate": (
            mixed_prompts / evaluable_prompts if evaluable_prompts else 0.0
        ),
        "stale_bytes_before_next_prompt": stale_bytes_before_next_prompt,
        "reusable_live_bytes": reusable_live_bytes,
        "total_live_bytes_in_evaluable_prompts": total_live_bytes_in_evaluable_prompts,
        "pinned_live_fraction": (
            reusable_live_bytes / total_live_bytes_in_evaluable_prompts
            if total_live_bytes_in_evaluable_prompts
            else 0.0
        ),
        "fragmentation_loss": (
            reusable_live_bytes / fragmentation_denominator
            if fragmentation_denominator
            else 0.0
        ),
        "avg_lifetime_spread": (
            statistics.fmean(lifetime_spreads) if lifetime_spreads else 0.0
        ),
        "lifetime_spread_per_prompt": lifetime_spreads,
    }


def _oracle_abstraction_bridge_summary(
    *,
    prompt_calls: Sequence[PromptCall],
    states: Mapping[str, StateRecord],
    next_prompt_ts_by_key: Mapping[tuple[str, str, str], int | None],
    next_prompt_call_by_key: Mapping[tuple[str, str, str], PromptCall | None],
    trace_end_ts: int,
) -> Dict[str, object]:
    monolithic_peak_hbm_bytes = 0
    ideal_segment_peak_hbm_bytes = 0
    monolithic_pinned_live_bytes = 0
    monolithic_stale_retained_bytes = 0
    monolithic_rematerialized_live_bytes = 0
    monolithic_reload_bytes = 0
    monolithic_recompute_bytes = 0
    monolithic_service_cost_units = 0.0
    ideal_segment_reload_bytes = 0
    ideal_segment_recompute_bytes = 0
    ideal_segment_service_cost_units = 0.0
    transition_count = 0

    for call in prompt_calls:
        key = (call.workflow_id, call.consumer, call.prompt_id)
        next_prompt_ts = next_prompt_ts_by_key.get(key)
        next_call = next_prompt_call_by_key.get(key)
        if next_prompt_ts is None or next_call is None:
            continue
        transition_count += 1

        current_ids = [state_id for state_id in call.state_ids if state_id in states]
        next_ids = [state_id for state_id in next_call.state_ids if state_id in states]
        current_set = set(current_ids)
        next_set = set(next_ids)

        current_bytes = sum(states[state_id].size_bytes for state_id in current_ids)
        next_bytes = sum(states[state_id].size_bytes for state_id in next_ids)
        overlap_ids = current_set & next_set
        stale_ids = current_set - next_set
        delta_next_ids = next_set - current_set
        overlap_bytes = sum(states[state_id].size_bytes for state_id in overlap_ids)
        stale_bytes = sum(states[state_id].size_bytes for state_id in stale_ids)
        delta_next_bytes = sum(states[state_id].size_bytes for state_id in delta_next_ids)

        monolithic_peak_hbm_bytes = max(monolithic_peak_hbm_bytes, current_bytes, next_bytes)
        ideal_segment_peak_hbm_bytes = max(
            ideal_segment_peak_hbm_bytes,
            current_bytes,
            next_bytes,
        )
        if overlap_ids and current_set != next_set:
            monolithic_peak_hbm_bytes = max(
                monolithic_peak_hbm_bytes,
                current_bytes + next_bytes,
            )
            ideal_segment_peak_hbm_bytes = max(
                ideal_segment_peak_hbm_bytes,
                overlap_bytes + delta_next_bytes,
            )

        live_states = []
        stale_live_states = []
        for state_id in current_ids:
            record = states[state_id]
            if record.end_ts(trace_end_ts) >= next_prompt_ts:
                live_states.append(record)
                if state_id not in next_set:
                    stale_live_states.append(record)

        if overlap_ids and current_set != next_set:
            monolithic_pinned_live_bytes += sum(record.size_bytes for record in live_states)
            monolithic_stale_retained_bytes += stale_bytes
            monolithic_rematerialized_live_bytes += overlap_bytes

        for state_id in next_ids:
            record = states[state_id]
            if record.materialization == "HBM":
                monolithic_recompute_bytes += record.size_bytes
                monolithic_service_cost_units += record.recompute_cost
            else:
                monolithic_reload_bytes += record.size_bytes
                monolithic_service_cost_units += record.reload_cost

        for state_id in delta_next_ids:
            record = states[state_id]
            if record.materialization == "HBM":
                ideal_segment_recompute_bytes += record.size_bytes
                ideal_segment_service_cost_units += record.recompute_cost
            else:
                ideal_segment_reload_bytes += record.size_bytes
                ideal_segment_service_cost_units += record.reload_cost

    return {
        "transition_count": transition_count,
        "monolithic_peak_hbm_bytes": monolithic_peak_hbm_bytes,
        "ideal_segment_peak_hbm_bytes": ideal_segment_peak_hbm_bytes,
        "monolithic_pinned_live_bytes": monolithic_pinned_live_bytes,
        "ideal_segment_pinned_live_bytes": 0,
        "monolithic_stale_retained_bytes": monolithic_stale_retained_bytes,
        "ideal_segment_stale_retained_bytes": 0,
        "monolithic_rematerialized_live_bytes": monolithic_rematerialized_live_bytes,
        "ideal_segment_rematerialized_live_bytes": 0,
        "monolithic_reload_bytes": monolithic_reload_bytes,
        "ideal_segment_reload_bytes": ideal_segment_reload_bytes,
        "monolithic_recompute_bytes": monolithic_recompute_bytes,
        "ideal_segment_recompute_bytes": ideal_segment_recompute_bytes,
        "monolithic_service_cost_units": monolithic_service_cost_units,
        "ideal_segment_service_cost_units": ideal_segment_service_cost_units,
        "oracle_peak_hbm_savings_fraction": (
            (monolithic_peak_hbm_bytes - ideal_segment_peak_hbm_bytes)
            / monolithic_peak_hbm_bytes
            if monolithic_peak_hbm_bytes
            else 0.0
        ),
        "oracle_pinned_live_savings_fraction": (
            1.0 if monolithic_pinned_live_bytes else 0.0
        ),
        "oracle_reload_savings_fraction": (
            (monolithic_reload_bytes - ideal_segment_reload_bytes) / monolithic_reload_bytes
            if monolithic_reload_bytes
            else 0.0
        ),
        "oracle_recompute_savings_fraction": (
            (monolithic_recompute_bytes - ideal_segment_recompute_bytes)
            / monolithic_recompute_bytes
            if monolithic_recompute_bytes
            else 0.0
        ),
        "oracle_rematerialization_savings_fraction": (
            1.0 if monolithic_rematerialized_live_bytes else 0.0
        ),
        "oracle_service_cost_savings_fraction": (
            (
                monolithic_service_cost_units - ideal_segment_service_cost_units
            )
            / monolithic_service_cost_units
            if monolithic_service_cost_units
            else 0.0
        ),
    }


def _abstraction_bridge_summary(
    *,
    prompt_calls: Sequence[PromptCall],
    states: Mapping[str, StateRecord],
    next_prompt_ts_by_key: Mapping[tuple[str, str, str], int | None],
    next_prompt_call_by_key: Mapping[tuple[str, str, str], PromptCall | None],
    trace_end_ts: int,
) -> Dict[str, object]:
    return _legacy_bridge_alias(
        _oracle_abstraction_bridge_summary(
            prompt_calls=prompt_calls,
            states=states,
            next_prompt_ts_by_key=next_prompt_ts_by_key,
            next_prompt_call_by_key=next_prompt_call_by_key,
            trace_end_ts=trace_end_ts,
        )
    )


def _empty_oracle_bridge_summary() -> Dict[str, object]:
    return {
        "transition_count": 0,
        "monolithic_peak_hbm_bytes": 0,
        "ideal_segment_peak_hbm_bytes": 0,
        "monolithic_pinned_live_bytes": 0,
        "ideal_segment_pinned_live_bytes": 0,
        "monolithic_stale_retained_bytes": 0,
        "ideal_segment_stale_retained_bytes": 0,
        "monolithic_rematerialized_live_bytes": 0,
        "ideal_segment_rematerialized_live_bytes": 0,
        "monolithic_reload_bytes": 0,
        "ideal_segment_reload_bytes": 0,
        "monolithic_recompute_bytes": 0,
        "ideal_segment_recompute_bytes": 0,
        "monolithic_service_cost_units": 0.0,
        "ideal_segment_service_cost_units": 0.0,
        "oracle_peak_hbm_savings_fraction": 0.0,
        "oracle_pinned_live_savings_fraction": 0.0,
        "oracle_reload_savings_fraction": 0.0,
        "oracle_recompute_savings_fraction": 0.0,
        "oracle_rematerialization_savings_fraction": 0.0,
        "oracle_service_cost_savings_fraction": 0.0,
    }


def _empty_runtime_behavior_summary() -> Dict[str, object]:
    return {
        "event_count": 0,
        "op_counts": {},
        "lookup_status_counts": {},
        "resident_hit_rate": 0.0,
        "registrations": 0,
        "resident_hits": 0,
        "evicted_hits": 0,
        "misses": 0,
        "invalid_lookups": 0,
        "materializations": 0,
        "rematerializations": 0,
        "reuse_count": 0,
        "reused_tokens": 0,
        "materialized_tokens": 0,
        "rematerialized_tokens": 0,
        "token_weighted_reuse_rate": 0.0,
        "reuse_benefit_ratio": 0.0,
        "prompt_token_reuse_fraction": 0.0,
        "lifecycle_reclaims": 0,
        "policy_reclaims": 0,
        "bytes_reclaimed": 0,
        "registered_segments": 0,
        "segment_reuse_buckets": {
            "registered_segments": 0,
            "registered_but_never_reused": 0,
            "reused_exactly_once": 0,
            "reused_more_than_five": 0,
        },
        "role_rows": [],
        "workflow_rows": [],
        "lifetime_rows": [],
        "segment_rows": [],
        "execution_context_rows": [],
    }


def _empty_reuse_locality_summary() -> Dict[str, object]:
    return {
        "segment_count": 0,
        "avg_accesses_per_segment": 0.0,
        "avg_revisits_per_segment": 0.0,
        "revisited_segment_fraction": 0.0,
        "avg_prompt_revisit_distance": 0.0,
        "module_rows": [],
    }


def _empty_backend_latency_summary() -> Dict[str, object]:
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


def _legacy_bridge_alias(oracle_bridge: Mapping[str, object]) -> Dict[str, object]:
    return {
        "transition_count": int(oracle_bridge["transition_count"]),
        "monolithic_peak_hbm_bytes": int(oracle_bridge["monolithic_peak_hbm_bytes"]),
        "segment_aware_peak_hbm_bytes": int(oracle_bridge["ideal_segment_peak_hbm_bytes"]),
        "monolithic_pinned_live_bytes": int(oracle_bridge["monolithic_pinned_live_bytes"]),
        "segment_aware_pinned_live_bytes": int(oracle_bridge["ideal_segment_pinned_live_bytes"]),
        "monolithic_stale_retained_bytes": int(oracle_bridge["monolithic_stale_retained_bytes"]),
        "segment_aware_stale_retained_bytes": int(oracle_bridge["ideal_segment_stale_retained_bytes"]),
        "monolithic_rematerialized_live_bytes": int(
            oracle_bridge["monolithic_rematerialized_live_bytes"]
        ),
        "segment_aware_rematerialized_live_bytes": int(
            oracle_bridge["ideal_segment_rematerialized_live_bytes"]
        ),
        "monolithic_reload_bytes": int(oracle_bridge["monolithic_reload_bytes"]),
        "segment_aware_reload_bytes": int(oracle_bridge["ideal_segment_reload_bytes"]),
        "monolithic_recompute_bytes": int(oracle_bridge["monolithic_recompute_bytes"]),
        "segment_aware_recompute_bytes": int(oracle_bridge["ideal_segment_recompute_bytes"]),
        "monolithic_service_cost_units": float(oracle_bridge["monolithic_service_cost_units"]),
        "segment_aware_service_cost_units": float(oracle_bridge["ideal_segment_service_cost_units"]),
        "peak_hbm_savings_fraction": float(oracle_bridge["oracle_peak_hbm_savings_fraction"]),
        "pinned_live_savings_fraction": float(
            oracle_bridge["oracle_pinned_live_savings_fraction"]
        ),
        "reload_savings_fraction": float(oracle_bridge["oracle_reload_savings_fraction"]),
        "recompute_savings_fraction": float(
            oracle_bridge["oracle_recompute_savings_fraction"]
        ),
        "rematerialization_savings_fraction": float(
            oracle_bridge["oracle_rematerialization_savings_fraction"]
        ),
        "service_cost_savings_fraction": float(
            oracle_bridge["oracle_service_cost_savings_fraction"]
        ),
    }


def _coerce_oracle_bridge(analysis: Mapping[str, object]) -> Dict[str, object]:
    if "oracle_abstraction_bridge" in analysis:
        return dict(analysis["oracle_abstraction_bridge"])

    legacy_bridge = analysis["abstraction_bridge"]
    return {
        "transition_count": int(legacy_bridge["transition_count"]),
        "monolithic_peak_hbm_bytes": int(legacy_bridge["monolithic_peak_hbm_bytes"]),
        "ideal_segment_peak_hbm_bytes": int(legacy_bridge["segment_aware_peak_hbm_bytes"]),
        "monolithic_pinned_live_bytes": int(legacy_bridge["monolithic_pinned_live_bytes"]),
        "ideal_segment_pinned_live_bytes": int(legacy_bridge["segment_aware_pinned_live_bytes"]),
        "monolithic_stale_retained_bytes": int(legacy_bridge["monolithic_stale_retained_bytes"]),
        "ideal_segment_stale_retained_bytes": int(legacy_bridge["segment_aware_stale_retained_bytes"]),
        "monolithic_rematerialized_live_bytes": int(
            legacy_bridge["monolithic_rematerialized_live_bytes"]
        ),
        "ideal_segment_rematerialized_live_bytes": int(
            legacy_bridge["segment_aware_rematerialized_live_bytes"]
        ),
        "monolithic_reload_bytes": int(legacy_bridge["monolithic_reload_bytes"]),
        "ideal_segment_reload_bytes": int(legacy_bridge["segment_aware_reload_bytes"]),
        "monolithic_recompute_bytes": int(legacy_bridge["monolithic_recompute_bytes"]),
        "ideal_segment_recompute_bytes": int(legacy_bridge["segment_aware_recompute_bytes"]),
        "monolithic_service_cost_units": float(legacy_bridge["monolithic_service_cost_units"]),
        "ideal_segment_service_cost_units": float(legacy_bridge["segment_aware_service_cost_units"]),
        "oracle_peak_hbm_savings_fraction": float(legacy_bridge["peak_hbm_savings_fraction"]),
        "oracle_pinned_live_savings_fraction": float(
            legacy_bridge["pinned_live_savings_fraction"]
        ),
        "oracle_reload_savings_fraction": float(legacy_bridge["reload_savings_fraction"]),
        "oracle_recompute_savings_fraction": float(legacy_bridge["recompute_savings_fraction"]),
        "oracle_rematerialization_savings_fraction": float(
            legacy_bridge["rematerialization_savings_fraction"]
        ),
        "oracle_service_cost_savings_fraction": float(
            legacy_bridge["service_cost_savings_fraction"]
        ),
    }


def _coerce_runtime_behavior(analysis: Mapping[str, object]) -> Dict[str, object]:
    return dict(analysis.get("runtime_behavior", _empty_runtime_behavior_summary()))


def _coerce_backend_latency(analysis: Mapping[str, object]) -> Dict[str, object]:
    return dict(analysis.get("backend_latency", _empty_backend_latency_summary()))


def _aggregate_runtime_behaviors(
    runtime_behaviors: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not runtime_behaviors:
        return _empty_runtime_behavior_summary()

    op_counts: Dict[str, int] = defaultdict(int)
    lookup_status_counts: Dict[str, int] = defaultdict(int)
    role_rows: Dict[str, Dict[str, object]] = {}
    workflow_rows: Dict[str, Dict[str, object]] = {}
    lifetime_rows: Dict[str, Dict[str, object]] = {}

    def ensure_row(
        rows: Dict[str, Dict[str, object]],
        key: str,
        *,
        row_key: str,
    ) -> Dict[str, object]:
        if key not in rows:
            rows[key] = {row_key: key}
        return rows[key]

    for behavior in runtime_behaviors:
        for op, count in dict(behavior.get("op_counts", {})).items():
            op_counts[str(op)] += int(count)
        for status, count in dict(behavior.get("lookup_status_counts", {})).items():
            lookup_status_counts[str(status)] += int(count)

        for row in behavior.get("role_rows", []):
            role = str(row["role"])
            output = ensure_row(role_rows, role, row_key="role")
            for field_name in (
                "registrations",
                "resident_hits",
                "evicted_hits",
                "misses",
                "invalid_lookups",
                "materializations",
                "rematerializations",
                "reuse_count",
                "lifecycle_reclaims",
                "policy_reclaims",
                "bytes_reclaimed",
                "reused_tokens",
                "materialized_tokens",
                "rematerialized_tokens",
                "registered_segments",
                "registered_but_never_reused",
                "reused_exactly_once",
                "reused_more_than_five",
            ):
                output[field_name] = int(output.get(field_name, 0)) + int(row[field_name])

        for row in behavior.get("workflow_rows", []):
            workflow_id = str(row["workflow_id"])
            output = ensure_row(workflow_rows, workflow_id, row_key="workflow_id")
            for field_name in (
                "registrations",
                "resident_hits",
                "evicted_hits",
                "misses",
                "invalid_lookups",
                "materializations",
                "rematerializations",
                "reuse_count",
                "lifecycle_reclaims",
                "policy_reclaims",
                "bytes_reclaimed",
                "reused_tokens",
                "materialized_tokens",
                "rematerialized_tokens",
            ):
                output[field_name] = int(output.get(field_name, 0)) + int(row[field_name])

        for row in behavior.get("lifetime_rows", []):
            role = str(row["role"])
            output = ensure_row(lifetime_rows, role, row_key="role")
            count = int(row["count"])
            output["count"] = int(output.get("count", 0)) + count
            for field_name in (
                "avg_semantic_lifetime",
                "avg_registration_to_first_lookup",
                "avg_lookup_span",
                "registered_but_never_reused_ratio",
            ):
                weighted_total = float(output.get(f"{field_name}_weighted_total", 0.0))
                weighted_total += float(row[field_name]) * count
                output[f"{field_name}_weighted_total"] = weighted_total

    resident_hits = lookup_status_counts.get("HIT_RESIDENT", 0)
    evicted_hits = lookup_status_counts.get("HIT_EVICTED", 0)
    misses = lookup_status_counts.get("MISS", 0)
    invalid_lookups = lookup_status_counts.get("INVALID", 0)
    denominator = resident_hits + evicted_hits + misses + invalid_lookups

    finalized_role_rows = []
    for role, row in sorted(role_rows.items()):
        row = dict(row)
        local_denominator = (
            int(row["resident_hits"])
            + int(row["evicted_hits"])
            + int(row["misses"])
            + int(row["invalid_lookups"])
        )
        row["resident_hit_rate"] = (
            int(row["resident_hits"]) / local_denominator if local_denominator else 0.0
        )
        row["token_weighted_reuse_rate"] = (
            int(row["reused_tokens"])
            / (int(row["reused_tokens"]) + int(row["materialized_tokens"]))
            if (int(row["reused_tokens"]) + int(row["materialized_tokens"]))
            else 0.0
        )
        finalized_role_rows.append(row)

    finalized_workflow_rows = []
    for workflow_id, row in sorted(workflow_rows.items()):
        row = dict(row)
        local_denominator = (
            int(row["resident_hits"])
            + int(row["evicted_hits"])
            + int(row["misses"])
            + int(row["invalid_lookups"])
        )
        row["resident_hit_rate"] = (
            int(row["resident_hits"]) / local_denominator if local_denominator else 0.0
        )
        row["token_weighted_reuse_rate"] = (
            int(row["reused_tokens"])
            / (int(row["reused_tokens"]) + int(row["materialized_tokens"]))
            if (int(row["reused_tokens"]) + int(row["materialized_tokens"]))
            else 0.0
        )
        finalized_workflow_rows.append(row)

    finalized_lifetime_rows = []
    for role, row in sorted(lifetime_rows.items()):
        count = int(row["count"])
        finalized_lifetime_rows.append(
            {
                "role": role,
                "count": count,
                "avg_semantic_lifetime": (
                    float(row.get("avg_semantic_lifetime_weighted_total", 0.0)) / count
                    if count
                    else 0.0
                ),
                "avg_registration_to_first_lookup": (
                    float(
                        row.get("avg_registration_to_first_lookup_weighted_total", 0.0)
                    )
                    / count
                    if count
                    else 0.0
                ),
                "avg_lookup_span": (
                    float(row.get("avg_lookup_span_weighted_total", 0.0)) / count
                    if count
                    else 0.0
                ),
                "registered_but_never_reused_ratio": (
                    float(
                        row.get(
                            "registered_but_never_reused_ratio_weighted_total",
                            0.0,
                        )
                    )
                    / count
                    if count
                    else 0.0
                ),
            }
        )

    registered_segments = sum(
        int(row["registered_segments"]) for row in finalized_role_rows
    )
    registered_but_never_reused = sum(
        int(row["registered_but_never_reused"]) for row in finalized_role_rows
    )
    reused_exactly_once = sum(
        int(row["reused_exactly_once"]) for row in finalized_role_rows
    )
    reused_more_than_five = sum(
        int(row["reused_more_than_five"]) for row in finalized_role_rows
    )
    reused_tokens = sum(int(row["reused_tokens"]) for row in finalized_role_rows)
    materialized_tokens = sum(int(row["materialized_tokens"]) for row in finalized_role_rows)
    rematerialized_tokens = sum(int(row["rematerialized_tokens"]) for row in finalized_role_rows)

    return {
        "event_count": sum(int(behavior["event_count"]) for behavior in runtime_behaviors),
        "op_counts": dict(sorted(op_counts.items())),
        "lookup_status_counts": dict(sorted(lookup_status_counts.items())),
        "resident_hit_rate": resident_hits / denominator if denominator else 0.0,
        "registrations": sum(int(behavior["registrations"]) for behavior in runtime_behaviors),
        "resident_hits": resident_hits,
        "evicted_hits": evicted_hits,
        "misses": misses,
        "invalid_lookups": invalid_lookups,
        "materializations": sum(
            int(behavior["materializations"]) for behavior in runtime_behaviors
        ),
        "rematerializations": sum(
            int(behavior["rematerializations"]) for behavior in runtime_behaviors
        ),
        "reuse_count": sum(int(behavior["reuse_count"]) for behavior in runtime_behaviors),
        "reused_tokens": reused_tokens,
        "materialized_tokens": materialized_tokens,
        "rematerialized_tokens": rematerialized_tokens,
        "token_weighted_reuse_rate": (
            reused_tokens / (reused_tokens + materialized_tokens)
            if (reused_tokens + materialized_tokens)
            else 0.0
        ),
        "reuse_benefit_ratio": (
            reused_tokens / (reused_tokens + materialized_tokens)
            if (reused_tokens + materialized_tokens)
            else 0.0
        ),
        "prompt_token_reuse_fraction": 0.0,
        "lifecycle_reclaims": sum(
            int(behavior["lifecycle_reclaims"]) for behavior in runtime_behaviors
        ),
        "policy_reclaims": sum(
            int(behavior["policy_reclaims"]) for behavior in runtime_behaviors
        ),
        "bytes_reclaimed": sum(
            int(behavior["bytes_reclaimed"]) for behavior in runtime_behaviors
        ),
        "registered_segments": registered_segments,
        "segment_reuse_buckets": {
            "registered_segments": registered_segments,
            "registered_but_never_reused": registered_but_never_reused,
            "reused_exactly_once": reused_exactly_once,
            "reused_more_than_five": reused_more_than_five,
        },
        "role_rows": finalized_role_rows,
        "workflow_rows": finalized_workflow_rows,
        "lifetime_rows": finalized_lifetime_rows,
        "segment_rows": [],
        "execution_context_rows": [],
    }


def _aggregate_backend_latencies(
    backend_latencies: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not backend_latencies:
        return _empty_backend_latency_summary()

    grouped: Dict[tuple[str, str], Dict[str, float | int | str]] = {}
    for latency in backend_latencies:
        for row in latency.get("step_rows", []):
            key = (str(row["step_name"]), str(row["prompt_mode"]))
            bucket = grouped.setdefault(
                key,
                {
                    "step_name": key[0],
                    "prompt_mode": key[1],
                    "request_count": 0,
                    "total_duration_ms": 0.0,
                    "total_backend_roundtrip_ms": 0.0,
                    "total_frontend_message_build_ms": 0.0,
                    "total_frontend_token_estimate_ms": 0.0,
                    "total_frontend_overhead_ms": 0.0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "frontend_cache_hits": 0,
                    "system_prompt_tokens_estimate": 0,
                    "user_prompt_tokens_estimate": 0,
                    "ordered_segment_tokens": 0,
                    "request_segment_tokens": 0,
                    "assembled_prompt_tokens_estimate": 0,
                    "prompt_payload_tokens_estimate": 0,
                    "duplicate_prompt_tokens_estimate": 0,
                },
            )
            bucket["request_count"] = int(bucket["request_count"]) + int(row["request_count"])
            bucket["total_duration_ms"] = float(bucket["total_duration_ms"]) + float(
                row["total_duration_ms"]
            )
            bucket["total_backend_roundtrip_ms"] = float(
                bucket["total_backend_roundtrip_ms"]
            ) + float(row.get("total_backend_roundtrip_ms", 0.0))
            bucket["total_frontend_message_build_ms"] = float(
                bucket["total_frontend_message_build_ms"]
            ) + float(row.get("total_frontend_message_build_ms", 0.0))
            bucket["total_frontend_token_estimate_ms"] = float(
                bucket["total_frontend_token_estimate_ms"]
            ) + float(row.get("total_frontend_token_estimate_ms", 0.0))
            bucket["total_frontend_overhead_ms"] = float(
                bucket["total_frontend_overhead_ms"]
            ) + float(row.get("total_frontend_overhead_ms", 0.0))
            bucket["total_prompt_tokens"] = int(bucket["total_prompt_tokens"]) + int(
                row["total_prompt_tokens"]
            )
            bucket["total_completion_tokens"] = int(
                bucket["total_completion_tokens"]
            ) + int(row["total_completion_tokens"])
            bucket["total_tokens"] = int(bucket["total_tokens"]) + int(row["total_tokens"])
            bucket["frontend_cache_hits"] = int(bucket["frontend_cache_hits"]) + int(
                row.get("frontend_cache_hits", 0)
            )
            bucket["system_prompt_tokens_estimate"] = int(
                bucket["system_prompt_tokens_estimate"]
            ) + int(row.get("system_prompt_tokens_estimate", 0))
            bucket["user_prompt_tokens_estimate"] = int(
                bucket["user_prompt_tokens_estimate"]
            ) + int(row.get("user_prompt_tokens_estimate", 0))
            bucket["ordered_segment_tokens"] = int(bucket["ordered_segment_tokens"]) + int(
                row.get("ordered_segment_tokens", 0)
            )
            bucket["request_segment_tokens"] = int(bucket["request_segment_tokens"]) + int(
                row.get("request_segment_tokens", 0)
            )
            bucket["assembled_prompt_tokens_estimate"] = int(
                bucket["assembled_prompt_tokens_estimate"]
            ) + int(row.get("assembled_prompt_tokens_estimate", 0))
            bucket["prompt_payload_tokens_estimate"] = int(
                bucket["prompt_payload_tokens_estimate"]
            ) + int(row.get("prompt_payload_tokens_estimate", 0))
            bucket["duplicate_prompt_tokens_estimate"] = int(
                bucket["duplicate_prompt_tokens_estimate"]
            ) + int(row.get("duplicate_prompt_tokens_estimate", 0))

    step_rows = []
    for key in sorted(grouped):
        bucket = grouped[key]
        request_count = int(bucket["request_count"])
        total_duration_ms = float(bucket["total_duration_ms"])
        total_prompt_tokens = int(bucket["total_prompt_tokens"])
        step_rows.append(
            {
                "step_name": bucket["step_name"],
                "prompt_mode": bucket["prompt_mode"],
                "request_count": request_count,
                "total_duration_ms": total_duration_ms,
                "avg_duration_ms": total_duration_ms / request_count if request_count else 0.0,
                "total_backend_roundtrip_ms": float(bucket["total_backend_roundtrip_ms"]),
                "avg_backend_roundtrip_ms": (
                    float(bucket["total_backend_roundtrip_ms"]) / request_count
                    if request_count
                    else 0.0
                ),
                "total_frontend_message_build_ms": float(
                    bucket["total_frontend_message_build_ms"]
                ),
                "avg_frontend_message_build_ms": (
                    float(bucket["total_frontend_message_build_ms"]) / request_count
                    if request_count
                    else 0.0
                ),
                "total_frontend_token_estimate_ms": float(
                    bucket["total_frontend_token_estimate_ms"]
                ),
                "avg_frontend_token_estimate_ms": (
                    float(bucket["total_frontend_token_estimate_ms"]) / request_count
                    if request_count
                    else 0.0
                ),
                "total_frontend_overhead_ms": float(bucket["total_frontend_overhead_ms"]),
                "avg_frontend_overhead_ms": (
                    float(bucket["total_frontend_overhead_ms"]) / request_count
                    if request_count
                    else 0.0
                ),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": int(bucket["total_completion_tokens"]),
                "total_tokens": int(bucket["total_tokens"]),
                "avg_prompt_tokens": (
                    total_prompt_tokens / request_count if request_count else 0.0
                ),
                "duration_ms_per_1k_prompt_tokens": (
                    total_duration_ms / (total_prompt_tokens / 1000.0)
                    if total_prompt_tokens
                    else 0.0
                ),
                "frontend_cache_hits": int(bucket["frontend_cache_hits"]),
                "frontend_cache_hit_rate": (
                    int(bucket["frontend_cache_hits"]) / request_count if request_count else 0.0
                ),
                "system_prompt_tokens_estimate": int(bucket["system_prompt_tokens_estimate"]),
                "user_prompt_tokens_estimate": int(bucket["user_prompt_tokens_estimate"]),
                "ordered_segment_tokens": int(bucket["ordered_segment_tokens"]),
                "request_segment_tokens": int(bucket["request_segment_tokens"]),
                "assembled_prompt_tokens_estimate": int(
                    bucket["assembled_prompt_tokens_estimate"]
                ),
                "prompt_payload_tokens_estimate": int(
                    bucket["prompt_payload_tokens_estimate"]
                ),
                "duplicate_prompt_tokens_estimate": int(
                    bucket["duplicate_prompt_tokens_estimate"]
                ),
            }
        )

    request_count = sum(int(latency["request_count"]) for latency in backend_latencies)
    total_duration_ms = sum(float(latency["total_duration_ms"]) for latency in backend_latencies)
    total_backend_roundtrip_ms = sum(
        float(latency.get("total_backend_roundtrip_ms", 0.0))
        for latency in backend_latencies
    )
    total_frontend_message_build_ms = sum(
        float(latency.get("total_frontend_message_build_ms", 0.0))
        for latency in backend_latencies
    )
    total_frontend_token_estimate_ms = sum(
        float(latency.get("total_frontend_token_estimate_ms", 0.0))
        for latency in backend_latencies
    )
    total_frontend_overhead_ms = sum(
        float(latency.get("total_frontend_overhead_ms", 0.0))
        for latency in backend_latencies
    )
    total_prompt_tokens = sum(int(latency["total_prompt_tokens"]) for latency in backend_latencies)
    total_completion_tokens = sum(
        int(latency["total_completion_tokens"]) for latency in backend_latencies
    )
    total_tokens = sum(int(latency["total_tokens"]) for latency in backend_latencies)
    frontend_cache_hits = sum(
        int(latency.get("frontend_cache_hits", 0)) for latency in backend_latencies
    )
    total_system_prompt_tokens_estimate = sum(
        int(latency.get("total_system_prompt_tokens_estimate", 0))
        for latency in backend_latencies
    )
    total_user_prompt_tokens_estimate = sum(
        int(latency.get("total_user_prompt_tokens_estimate", 0))
        for latency in backend_latencies
    )
    total_ordered_segment_tokens = sum(
        int(latency.get("total_ordered_segment_tokens", 0))
        for latency in backend_latencies
    )
    total_request_segment_tokens = sum(
        int(latency.get("total_request_segment_tokens", 0))
        for latency in backend_latencies
    )
    total_assembled_prompt_tokens_estimate = sum(
        int(latency.get("total_assembled_prompt_tokens_estimate", 0))
        for latency in backend_latencies
    )
    total_prompt_payload_tokens_estimate = sum(
        int(latency.get("total_prompt_payload_tokens_estimate", 0))
        for latency in backend_latencies
    )
    total_duplicate_prompt_tokens_estimate = sum(
        int(latency.get("total_duplicate_prompt_tokens_estimate", 0))
        for latency in backend_latencies
    )

    return {
        "request_count": request_count,
        "total_duration_ms": total_duration_ms,
        "avg_duration_ms": total_duration_ms / request_count if request_count else 0.0,
        "total_backend_roundtrip_ms": total_backend_roundtrip_ms,
        "avg_backend_roundtrip_ms": (
            total_backend_roundtrip_ms / request_count if request_count else 0.0
        ),
        "total_frontend_message_build_ms": total_frontend_message_build_ms,
        "avg_frontend_message_build_ms": (
            total_frontend_message_build_ms / request_count if request_count else 0.0
        ),
        "total_frontend_token_estimate_ms": total_frontend_token_estimate_ms,
        "avg_frontend_token_estimate_ms": (
            total_frontend_token_estimate_ms / request_count if request_count else 0.0
        ),
        "total_frontend_overhead_ms": total_frontend_overhead_ms,
        "avg_frontend_overhead_ms": (
            total_frontend_overhead_ms / request_count if request_count else 0.0
        ),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens": (
            total_prompt_tokens / request_count if request_count else 0.0
        ),
        "duration_ms_per_1k_prompt_tokens": (
            total_duration_ms / (total_prompt_tokens / 1000.0)
            if total_prompt_tokens
            else 0.0
        ),
        "frontend_cache_hits": frontend_cache_hits,
        "frontend_cache_hit_rate": (
            frontend_cache_hits / request_count if request_count else 0.0
        ),
        "total_system_prompt_tokens_estimate": total_system_prompt_tokens_estimate,
        "total_user_prompt_tokens_estimate": total_user_prompt_tokens_estimate,
        "total_ordered_segment_tokens": total_ordered_segment_tokens,
        "total_request_segment_tokens": total_request_segment_tokens,
        "total_assembled_prompt_tokens_estimate": total_assembled_prompt_tokens_estimate,
        "total_prompt_payload_tokens_estimate": total_prompt_payload_tokens_estimate,
        "total_duplicate_prompt_tokens_estimate": total_duplicate_prompt_tokens_estimate,
        "step_rows": step_rows,
    }


def _aggregate_analyses(analyses: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not analyses:
        return {
            "state_count": 0,
            "prompt_call_count": 0,
            "prompt_composition": {"module_rows": []},
            "lifecycle_characterization": {"module_rows": []},
            "lifetime_correlation": _lifetime_correlation_summary_from_vectors([]),
            "abstraction_mismatch": {
                "evaluable_prompt_count": 0,
                "monolithic_invalidation_events": 0,
                "mixed_lifecycle_prompt_rate": 0.0,
                "stale_bytes_before_next_prompt": 0,
                "reusable_live_bytes": 0,
                "total_live_bytes_in_evaluable_prompts": 0,
                "pinned_live_fraction": 0.0,
                "fragmentation_loss": 0.0,
                "avg_lifetime_spread": 0.0,
                "lifetime_spread_per_prompt": [],
            },
            "reuse_locality": _empty_reuse_locality_summary(),
            "runtime_behavior": _empty_runtime_behavior_summary(),
            "backend_latency": _empty_backend_latency_summary(),
            "oracle_abstraction_bridge": _empty_oracle_bridge_summary(),
            "abstraction_bridge": _legacy_bridge_alias(_empty_oracle_bridge_summary()),
        }

    total_state_count = sum(int(analysis["state_count"]) for analysis in analyses)
    total_prompt_calls = sum(int(analysis["prompt_call_count"]) for analysis in analyses)

    module_prompt_rows = _average_rows(
        [analysis["prompt_composition"]["module_rows"] for analysis in analyses],
        [
            "prompt_presence_rate",
            "avg_segments_per_prompt",
            "avg_bytes_per_prompt",
            "immutable_fraction",
            "shared_fraction",
            "ephemeral_fraction",
        ],
    )
    module_lifecycle_rows = _average_rows(
        [analysis["lifecycle_characterization"]["module_rows"] for analysis in analyses],
        [
            "avg_lifetime",
            "median_lifetime",
            "avg_reads",
            "superseded_fraction",
            "released_fraction",
        ],
        count_field="count",
    )
    lifetime_correlation = _lifetime_correlation_summary_from_vectors(
        [
            vector
            for analysis in analyses
            for vector in analysis["lifetime_correlation"]["prompt_vectors"]
        ]
    )

    mismatch = {
        "evaluable_prompt_count": sum(
            int(analysis["abstraction_mismatch"]["evaluable_prompt_count"])
            for analysis in analyses
        ),
        "monolithic_invalidation_events": sum(
            int(analysis["abstraction_mismatch"]["monolithic_invalidation_events"])
            for analysis in analyses
        ),
        "stale_bytes_before_next_prompt": sum(
            int(analysis["abstraction_mismatch"]["stale_bytes_before_next_prompt"])
            for analysis in analyses
        ),
        "reusable_live_bytes": sum(
            int(analysis["abstraction_mismatch"]["reusable_live_bytes"])
            for analysis in analyses
        ),
        "total_live_bytes_in_evaluable_prompts": sum(
            int(analysis["abstraction_mismatch"]["total_live_bytes_in_evaluable_prompts"])
            for analysis in analyses
        ),
        "avg_lifetime_spread": statistics.fmean(
            analysis["abstraction_mismatch"]["avg_lifetime_spread"] for analysis in analyses
        ),
        "lifetime_spread_per_prompt": [
            spread
            for analysis in analyses
            for spread in analysis["abstraction_mismatch"]["lifetime_spread_per_prompt"]
        ],
    }
    mismatch["mixed_lifecycle_prompt_rate"] = (
        mismatch["monolithic_invalidation_events"] / mismatch["evaluable_prompt_count"]
        if mismatch["evaluable_prompt_count"]
        else 0.0
    )
    mismatch["pinned_live_fraction"] = (
        mismatch["reusable_live_bytes"] / mismatch["total_live_bytes_in_evaluable_prompts"]
        if mismatch["total_live_bytes_in_evaluable_prompts"]
        else 0.0
    )
    mismatch["fragmentation_loss"] = (
        mismatch["reusable_live_bytes"]
        / (mismatch["reusable_live_bytes"] + mismatch["stale_bytes_before_next_prompt"])
        if (mismatch["reusable_live_bytes"] + mismatch["stale_bytes_before_next_prompt"])
        else 0.0
    )

    reuse_locality = _aggregate_reuse_locality_summaries(
        [
            dict(analysis.get("reuse_locality", _empty_reuse_locality_summary()))
            for analysis in analyses
        ]
    )

    oracle_bridges = [_coerce_oracle_bridge(analysis) for analysis in analyses]
    oracle_abstraction_bridge = {
        "transition_count": sum(
            int(bridge["transition_count"]) for bridge in oracle_bridges
        ),
        "monolithic_peak_hbm_bytes": max(
            int(bridge["monolithic_peak_hbm_bytes"]) for bridge in oracle_bridges
        ),
        "ideal_segment_peak_hbm_bytes": max(
            int(bridge["ideal_segment_peak_hbm_bytes"]) for bridge in oracle_bridges
        ),
        "monolithic_pinned_live_bytes": sum(
            int(bridge["monolithic_pinned_live_bytes"]) for bridge in oracle_bridges
        ),
        "ideal_segment_pinned_live_bytes": sum(
            int(bridge["ideal_segment_pinned_live_bytes"]) for bridge in oracle_bridges
        ),
        "monolithic_stale_retained_bytes": sum(
            int(bridge["monolithic_stale_retained_bytes"]) for bridge in oracle_bridges
        ),
        "ideal_segment_stale_retained_bytes": sum(
            int(bridge["ideal_segment_stale_retained_bytes"]) for bridge in oracle_bridges
        ),
        "monolithic_rematerialized_live_bytes": sum(
            int(bridge["monolithic_rematerialized_live_bytes"]) for bridge in oracle_bridges
        ),
        "ideal_segment_rematerialized_live_bytes": sum(
            int(bridge["ideal_segment_rematerialized_live_bytes"]) for bridge in oracle_bridges
        ),
        "monolithic_reload_bytes": sum(
            int(bridge["monolithic_reload_bytes"]) for bridge in oracle_bridges
        ),
        "ideal_segment_reload_bytes": sum(
            int(bridge["ideal_segment_reload_bytes"]) for bridge in oracle_bridges
        ),
        "monolithic_recompute_bytes": sum(
            int(bridge["monolithic_recompute_bytes"]) for bridge in oracle_bridges
        ),
        "ideal_segment_recompute_bytes": sum(
            int(bridge["ideal_segment_recompute_bytes"]) for bridge in oracle_bridges
        ),
        "monolithic_service_cost_units": sum(
            float(bridge["monolithic_service_cost_units"]) for bridge in oracle_bridges
        ),
        "ideal_segment_service_cost_units": sum(
            float(bridge["ideal_segment_service_cost_units"]) for bridge in oracle_bridges
        ),
    }
    oracle_abstraction_bridge["oracle_peak_hbm_savings_fraction"] = (
        (
            oracle_abstraction_bridge["monolithic_peak_hbm_bytes"]
            - oracle_abstraction_bridge["ideal_segment_peak_hbm_bytes"]
        )
        / oracle_abstraction_bridge["monolithic_peak_hbm_bytes"]
        if oracle_abstraction_bridge["monolithic_peak_hbm_bytes"]
        else 0.0
    )
    oracle_abstraction_bridge["oracle_pinned_live_savings_fraction"] = (
        (
            oracle_abstraction_bridge["monolithic_pinned_live_bytes"]
            - oracle_abstraction_bridge["ideal_segment_pinned_live_bytes"]
        )
        / oracle_abstraction_bridge["monolithic_pinned_live_bytes"]
        if oracle_abstraction_bridge["monolithic_pinned_live_bytes"]
        else 0.0
    )
    oracle_abstraction_bridge["oracle_reload_savings_fraction"] = (
        (
            oracle_abstraction_bridge["monolithic_reload_bytes"]
            - oracle_abstraction_bridge["ideal_segment_reload_bytes"]
        )
        / oracle_abstraction_bridge["monolithic_reload_bytes"]
        if oracle_abstraction_bridge["monolithic_reload_bytes"]
        else 0.0
    )
    oracle_abstraction_bridge["oracle_recompute_savings_fraction"] = (
        (
            oracle_abstraction_bridge["monolithic_recompute_bytes"]
            - oracle_abstraction_bridge["ideal_segment_recompute_bytes"]
        )
        / oracle_abstraction_bridge["monolithic_recompute_bytes"]
        if oracle_abstraction_bridge["monolithic_recompute_bytes"]
        else 0.0
    )
    oracle_abstraction_bridge["oracle_rematerialization_savings_fraction"] = (
        (
            oracle_abstraction_bridge["monolithic_rematerialized_live_bytes"]
            - oracle_abstraction_bridge["ideal_segment_rematerialized_live_bytes"]
        )
        / oracle_abstraction_bridge["monolithic_rematerialized_live_bytes"]
        if oracle_abstraction_bridge["monolithic_rematerialized_live_bytes"]
        else 0.0
    )
    oracle_abstraction_bridge["oracle_service_cost_savings_fraction"] = (
        (
            oracle_abstraction_bridge["monolithic_service_cost_units"]
            - oracle_abstraction_bridge["ideal_segment_service_cost_units"]
        )
        / oracle_abstraction_bridge["monolithic_service_cost_units"]
        if oracle_abstraction_bridge["monolithic_service_cost_units"]
        else 0.0
    )

    runtime_behavior = _aggregate_runtime_behaviors(
        [
            _coerce_runtime_behavior(analysis)
            for analysis in analyses
            if "runtime_behavior" in analysis
        ]
    )
    backend_latency = _aggregate_backend_latencies(
        [
            _coerce_backend_latency(analysis)
            for analysis in analyses
            if "backend_latency" in analysis
        ]
    )
    materialized = int(runtime_behavior.get("materialized_tokens", 0))
    reused = int(runtime_behavior.get("reused_tokens", 0))
    runtime_behavior["reuse_benefit_ratio"] = (
        reused / (reused + materialized) if (reused + materialized) else 0.0
    )
    total_prompt_tokens = int(backend_latency.get("total_prompt_tokens", 0))
    runtime_behavior["prompt_token_reuse_fraction"] = (
        reused / total_prompt_tokens if total_prompt_tokens else 0.0
    )

    return {
        "state_count": total_state_count,
        "prompt_call_count": total_prompt_calls,
        "prompt_composition": {"module_rows": module_prompt_rows},
        "lifecycle_characterization": {"module_rows": module_lifecycle_rows},
        "lifetime_correlation": lifetime_correlation,
        "abstraction_mismatch": mismatch,
        "reuse_locality": reuse_locality,
        "runtime_behavior": runtime_behavior,
        "backend_latency": backend_latency,
        "oracle_abstraction_bridge": oracle_abstraction_bridge,
        "abstraction_bridge": _legacy_bridge_alias(oracle_abstraction_bridge),
    }


def _reuse_locality_summary(
    *,
    prompt_calls: Sequence[PromptCall],
    states: Mapping[str, StateRecord],
) -> Dict[str, object]:
    if not states:
        return _empty_reuse_locality_summary()

    prompt_index_by_prompt_id: Dict[str, int] = {}
    for index, call in enumerate(
        sorted(prompt_calls, key=lambda call: (call.ts, call.consumer, call.prompt_id))
    ):
        prompt_index_by_prompt_id.setdefault(call.prompt_id, index)
    module_buckets: Dict[str, Dict[str, float | int | str | List[float]]] = {}

    def ensure_bucket(module: str) -> Dict[str, float | int | str | List[float]]:
        if module not in module_buckets:
            module_buckets[module] = {
                "module": module,
                "count": 0,
                "accesses_total": 0,
                "revisits_total": 0,
                "revisited_count": 0,
                "prompt_revisit_distances": [],
            }
        return module_buckets[module]

    all_access_counts: List[int] = []
    all_revisit_counts: List[int] = []
    all_prompt_revisit_distances: List[float] = []
    revisited_segments = 0

    for state in states.values():
        access_count = len(state.read_prompt_ids)
        revisit_count = max(0, access_count - 1)
        all_access_counts.append(access_count)
        all_revisit_counts.append(revisit_count)
        if revisit_count > 0:
            revisited_segments += 1

        prompt_indices = []
        seen_indices = set()
        for prompt_id in state.read_prompt_ids:
            prompt_index = prompt_index_by_prompt_id.get(prompt_id)
            if prompt_index is None or prompt_index in seen_indices:
                continue
            seen_indices.add(prompt_index)
            prompt_indices.append(prompt_index)
        distances = [
            float(prompt_indices[index] - prompt_indices[index - 1])
            for index in range(1, len(prompt_indices))
            if prompt_indices[index] > prompt_indices[index - 1]
        ]
        all_prompt_revisit_distances.extend(distances)

        bucket = ensure_bucket(state.module)
        bucket["count"] = int(bucket["count"]) + 1
        bucket["accesses_total"] = int(bucket["accesses_total"]) + access_count
        bucket["revisits_total"] = int(bucket["revisits_total"]) + revisit_count
        if revisit_count > 0:
            bucket["revisited_count"] = int(bucket["revisited_count"]) + 1
        cast_distances = list(bucket["prompt_revisit_distances"])
        cast_distances.extend(distances)
        bucket["prompt_revisit_distances"] = cast_distances

    module_rows = []
    for module, bucket in sorted(module_buckets.items()):
        count = int(bucket["count"])
        accesses_total = int(bucket["accesses_total"])
        revisits_total = int(bucket["revisits_total"])
        revisited_count = int(bucket["revisited_count"])
        distances = [float(value) for value in bucket["prompt_revisit_distances"]]
        module_rows.append(
            {
                "module": module,
                "count": count,
                "avg_accesses_per_segment": accesses_total / count if count else 0.0,
                "avg_revisits_per_segment": revisits_total / count if count else 0.0,
                "revisited_segment_fraction": revisited_count / count if count else 0.0,
                "avg_prompt_revisit_distance": (
                    statistics.fmean(distances) if distances else 0.0
                ),
            }
        )

    segment_count = len(states)
    return {
        "segment_count": segment_count,
        "avg_accesses_per_segment": (
            statistics.fmean(all_access_counts) if all_access_counts else 0.0
        ),
        "avg_revisits_per_segment": (
            statistics.fmean(all_revisit_counts) if all_revisit_counts else 0.0
        ),
        "revisited_segment_fraction": (
            revisited_segments / segment_count if segment_count else 0.0
        ),
        "avg_prompt_revisit_distance": (
            statistics.fmean(all_prompt_revisit_distances)
            if all_prompt_revisit_distances
            else 0.0
        ),
        "module_rows": module_rows,
    }


def _aggregate_reuse_locality_summaries(
    reuse_localities: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not reuse_localities:
        return _empty_reuse_locality_summary()

    module_rows = _average_rows(
        [locality.get("module_rows", []) for locality in reuse_localities],
        [
            "avg_accesses_per_segment",
            "avg_revisits_per_segment",
            "revisited_segment_fraction",
            "avg_prompt_revisit_distance",
        ],
        count_field="count",
    )
    segment_count = sum(int(locality.get("segment_count", 0)) for locality in reuse_localities)
    accesses_weighted = sum(
        float(locality.get("avg_accesses_per_segment", 0.0)) * int(locality.get("segment_count", 0))
        for locality in reuse_localities
    )
    revisits_weighted = sum(
        float(locality.get("avg_revisits_per_segment", 0.0)) * int(locality.get("segment_count", 0))
        for locality in reuse_localities
    )
    revisited_fraction_weighted = sum(
        float(locality.get("revisited_segment_fraction", 0.0)) * int(locality.get("segment_count", 0))
        for locality in reuse_localities
    )
    distances = [
        float(locality.get("avg_prompt_revisit_distance", 0.0))
        for locality in reuse_localities
        if float(locality.get("avg_prompt_revisit_distance", 0.0)) > 0.0
    ]
    return {
        "segment_count": segment_count,
        "avg_accesses_per_segment": accesses_weighted / segment_count if segment_count else 0.0,
        "avg_revisits_per_segment": revisits_weighted / segment_count if segment_count else 0.0,
        "revisited_segment_fraction": (
            revisited_fraction_weighted / segment_count if segment_count else 0.0
        ),
        "avg_prompt_revisit_distance": (
            statistics.fmean(distances) if distances else 0.0
        ),
        "module_rows": module_rows,
    }


def _average_rows(
    row_sets: Sequence[Sequence[Mapping[str, object]]],
    numeric_fields: Sequence[str],
    count_field: str | None = None,
) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rows in row_sets:
        for row in rows:
            module = str(row["module"])
            for field_name in numeric_fields:
                grouped[module][field_name].append(float(row[field_name]))
            if count_field is not None:
                grouped[module][count_field].append(float(row[count_field]))

    output = []
    for module in sorted(grouped):
        payload: Dict[str, object] = {"module": module}
        for field_name in numeric_fields:
            payload[field_name] = statistics.fmean(grouped[module][field_name])
        if count_field is not None:
            payload[count_field] = int(
                round(statistics.fmean(grouped[module][count_field]))
            )
        output.append(payload)
    return output


def _lifetime_correlation_summary_from_vectors(
    prompt_vectors: Sequence[Mapping[str, float]],
) -> Dict[str, object]:
    module_order = sorted({module for vector in prompt_vectors for module in vector})
    correlation_matrix = []
    covariance_matrix = []
    heatmap_matrix = []
    pair_count_matrix = []

    for left_module in module_order:
        correlation_row = []
        covariance_row = []
        heatmap_row = []
        pair_count_row = []
        for right_module in module_order:
            left_values = []
            right_values = []
            for vector in prompt_vectors:
                if left_module in vector and right_module in vector:
                    left_values.append(float(vector[left_module]))
                    right_values.append(float(vector[right_module]))
            pair_count_row.append(len(left_values))
            covariance = _sample_covariance(left_values, right_values)
            if left_module == right_module and left_values:
                correlation = 1.0
            else:
                correlation = _sample_correlation(left_values, right_values)
            correlation_row.append(correlation)
            covariance_row.append(covariance)
            heatmap_row.append(_heatmap_bucket(correlation))
        correlation_matrix.append(correlation_row)
        covariance_matrix.append(covariance_row)
        heatmap_matrix.append(heatmap_row)
        pair_count_matrix.append(pair_count_row)

    return {
        "module_order": module_order,
        "prompt_vector_count": len(prompt_vectors),
        "prompt_vectors": [dict(vector) for vector in prompt_vectors],
        "correlation_matrix": correlation_matrix,
        "covariance_matrix": covariance_matrix,
        "heatmap_matrix": heatmap_matrix,
        "pair_count_matrix": pair_count_matrix,
    }


def _sample_covariance(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    return sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    ) / (len(left) - 1)


def _sample_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    covariance = _sample_covariance(left, right)
    if covariance is None:
        return None
    left_variance = _sample_covariance(left, left)
    right_variance = _sample_covariance(right, right)
    if (
        left_variance is None
        or right_variance is None
        or left_variance <= 0
        or right_variance <= 0
    ):
        return None
    return covariance / math.sqrt(left_variance * right_variance)


def _heatmap_bucket(value: float | None) -> str:
    if value is None:
        return "."
    if value >= 0.75:
        return "++"
    if value >= 0.25:
        return "+"
    if value <= -0.75:
        return "--"
    if value <= -0.25:
        return "-"
    return "0"


def _render_matrix_markdown(
    module_order: Sequence[str],
    matrix: Sequence[Sequence[object]],
    formatter,
) -> List[str]:
    if not module_order:
        return ["No prompt vectors available."]
    lines = [
        "| Module | " + " | ".join(module_order) + " |",
        "| - | " + " | ".join(["-:" for _ in module_order]) + " |",
    ]
    for module, row in zip(module_order, matrix):
        lines.append(
            "| " + module + " | " + " | ".join(formatter(value) for value in row) + " |"
        )
    return lines


def _format_matrix_value(value: object, *, precision: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"
