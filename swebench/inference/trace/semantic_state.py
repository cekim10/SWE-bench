from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence

STATE_TYPES = {
    "agent_anchor",
    "conversation_history",
    "plan",
    "tool_input",
    "tool_output",
    "retrieved_document",
    "generated_artifact",
    "error_diagnostic",
    "summary",
    "verification_result",
    "scratch",
}

OWNER_SCOPES = {"invocation", "agent", "workflow", "tenant", "shared"}
TIERS = {"HBM", "CPU", "DISK", "NONE"}

DEFAULT_RECOMPUTE_COST_DIVISOR = 16.0
DEFAULT_RELOAD_COST_DIVISOR = 256.0


class TraceLoggerError(ValueError):
    """Raised when a semantic state trace event is invalid."""


@dataclass(frozen=True)
class StateHandle:
    state_id: str
    logical_key: str
    version: int
    state_type: str


@dataclass
class _StateRecord:
    handle: StateHandle
    producer: str
    owner_scope: str


def get_schema_path() -> Path:
    return Path(__file__).with_name("semantic_state_trace.schema.json")


class TraceLogger:
    """Write semantic state lifecycle events as JSONL.

    This logger is designed for multi-step agent runners that need to emit:
    - state creation and derivation
    - prompt assembly reads
    - version transitions
    - materialization events
    """

    def __init__(
        self,
        trace_path: str | Path,
        *,
        workflow_id: str,
        tenant_id: str,
        default_owner_scope: str = "workflow",
        clock: Callable[[], int] | None = None,
        auto_flush: bool = True,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.workflow_id = workflow_id
        self.tenant_id = tenant_id
        self.default_owner_scope = self._normalize_owner_scope(default_owner_scope)
        self._clock = clock
        self._auto_flush = auto_flush
        self._last_ts = -1
        self._lock = threading.Lock()
        self._states: Dict[str, _StateRecord] = {}
        self._latest_state_by_key: Dict[str, str] = {}
        self._latest_version_by_key: Dict[str, int] = {}
        # Each workflow run should produce a standalone trace file.
        self._handle = self.trace_path.open("w", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def create_state(
        self,
        *,
        state_id: str,
        logical_key: str,
        state_type: str,
        size_bytes: int,
        token_count: int,
        producer: str,
        owner_scope: str | None = None,
        version: int | None = None,
        parent_state_ids: Sequence[str] | None = None,
        supersedes: str | None = None,
        recompute_cost: float | None = None,
        reload_cost: float | None = None,
        materialization: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> StateHandle:
        with self._lock:
            self._validate_new_state_id(state_id)
            normalized_state_type = self._normalize_state_type(state_type)
            normalized_owner_scope = self._normalize_owner_scope(owner_scope or self.default_owner_scope)
            normalized_materialization = (
                self._normalize_tier(materialization) if materialization is not None else None
            )
            validated_parents = self._normalize_parent_state_ids(parent_state_ids)
            resolved_version = version or (self._latest_version_by_key.get(logical_key, 0) + 1)
            if resolved_version < 1:
                raise TraceLoggerError(f"version must be >= 1, got {resolved_version}")
            resolved_supersedes = supersedes
            if resolved_supersedes is None and logical_key in self._latest_state_by_key and resolved_version > 1:
                resolved_supersedes = self._latest_state_by_key[logical_key]
            if resolved_supersedes is not None and resolved_supersedes not in self._states:
                raise TraceLoggerError(f"unknown superseded state_id {resolved_supersedes!r}")

            payload = self._base_payload(metadata)
            payload.update(
                {
                    "state_id": state_id,
                    "logical_key": logical_key,
                    "state_type": normalized_state_type,
                    "size_bytes": self._positive_int("size_bytes", size_bytes),
                    "token_count": self._positive_int("token_count", token_count),
                    "producer": producer,
                    "owner_scope": normalized_owner_scope,
                    "version": resolved_version,
                    "recompute_cost": self._non_negative_float(
                        "recompute_cost",
                        recompute_cost if recompute_cost is not None else max(1.0, token_count / DEFAULT_RECOMPUTE_COST_DIVISOR),
                    ),
                    "reload_cost": self._non_negative_float(
                        "reload_cost",
                        reload_cost if reload_cost is not None else max(0.5, size_bytes / DEFAULT_RELOAD_COST_DIVISOR),
                    ),
                }
            )
            if normalized_materialization is not None:
                payload["materialization"] = normalized_materialization
            if resolved_supersedes is not None:
                payload["supersedes"] = resolved_supersedes
            if validated_parents:
                payload["parent_state_ids"] = validated_parents

            handle = StateHandle(
                state_id=state_id,
                logical_key=logical_key,
                version=resolved_version,
                state_type=normalized_state_type,
            )
            self._states[state_id] = _StateRecord(
                handle=handle,
                producer=producer,
                owner_scope=normalized_owner_scope,
            )
            self._latest_state_by_key[logical_key] = state_id
            self._latest_version_by_key[logical_key] = max(
                resolved_version,
                self._latest_version_by_key.get(logical_key, 0),
            )
            self._write_event("DERIVE" if validated_parents else "CREATE", payload)
            return handle

    def share_state(
        self,
        state_id: str,
        *,
        consumer: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._write_state_reference_event("SHARE", state_id=state_id, consumer=consumer, metadata=metadata)

    def read_state(
        self,
        state_id: str,
        *,
        consumer: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._write_state_reference_event("READ", state_id=state_id, consumer=consumer, metadata=metadata)

    def read_states(
        self,
        state_ids: Iterable[str],
        *,
        consumer: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        for state_id in state_ids:
            self.read_state(state_id, consumer=consumer, metadata=metadata)

    def log_prompt_segments(
        self,
        *,
        consumer: str,
        prompt_id: str,
        segments: Iterable[Mapping[str, object]],
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        for position, segment in enumerate(segments):
            state_id = str(segment["state_id"])
            segment_metadata = dict(metadata or {})
            segment_metadata["prompt_id"] = prompt_id
            segment_metadata["segment_position"] = position
            for key, value in segment.items():
                if key == "state_id":
                    continue
                segment_metadata[key] = value
            self.read_state(
                state_id,
                consumer=consumer,
                metadata=segment_metadata,
            )

    def log_prompt_assembly(
        self,
        *,
        consumer: str,
        state_ids: Iterable[str],
        prompt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        segments = [{"state_id": state_id} for state_id in state_ids]
        self.log_prompt_segments(
            consumer=consumer,
            prompt_id=prompt_id or f"{consumer}-prompt",
            segments=segments,
            metadata=metadata,
        )

    def release_state(
        self,
        state_id: str,
        *,
        consumer: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._write_state_reference_event("RELEASE", state_id=state_id, consumer=consumer, metadata=metadata)

    def release_states(
        self,
        state_ids: Iterable[str],
        *,
        consumer: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        for state_id in state_ids:
            self.release_state(state_id, consumer=consumer, metadata=metadata)

    def supersede_state(
        self,
        old_state_id: str,
        new_state_id: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._require_state(old_state_id)
            self._require_state(new_state_id)
            payload = self._base_payload(metadata)
            payload.update(
                {
                    "old_state_id": old_state_id,
                    "new_state_id": new_state_id,
                }
            )
            self._write_event("SUPERSEDE", payload)

    def materialize_state(
        self,
        state_id: str,
        *,
        tier: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._require_state(state_id)
            payload = self._base_payload(metadata)
            payload.update(
                {
                    "state_id": state_id,
                    "tier": self._normalize_tier(tier),
                }
            )
            self._write_event("MATERIALIZE", payload)

    def evict_state(
        self,
        state_id: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._require_state(state_id)
            payload = self._base_payload(metadata)
            payload["state_id"] = state_id
            self._write_event("EVICT", payload)

    def reload_state(
        self,
        state_id: str,
        *,
        from_tier: str,
        to_tier: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._require_state(state_id)
            payload = self._base_payload(metadata)
            payload.update(
                {
                    "state_id": state_id,
                    "from_tier": self._normalize_tier(from_tier),
                    "to_tier": self._normalize_tier(to_tier),
                }
            )
            self._write_event("RELOAD", payload)

    def latest_state_id(self, logical_key: str) -> str | None:
        return self._latest_state_by_key.get(logical_key)

    def latest_version(self, logical_key: str) -> int:
        return self._latest_version_by_key.get(logical_key, 0)

    def _write_state_reference_event(
        self,
        op: str,
        *,
        state_id: str,
        consumer: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._require_state(state_id)
            payload = self._base_payload(metadata)
            payload.update({"state_id": state_id, "consumer": consumer})
            self._write_event(op, payload)

    def _validate_new_state_id(self, state_id: str) -> None:
        if state_id in self._states:
            raise TraceLoggerError(f"duplicate state_id {state_id!r}")

    def _require_state(self, state_id: str) -> None:
        if state_id not in self._states:
            raise TraceLoggerError(f"unknown state_id {state_id!r}")

    def _normalize_parent_state_ids(self, parent_state_ids: Sequence[str] | None) -> list[str]:
        if parent_state_ids is None:
            return []
        normalized = list(parent_state_ids)
        if not normalized:
            raise TraceLoggerError("parent_state_ids must be non-empty when provided")
        for parent_state_id in normalized:
            self._require_state(parent_state_id)
        return normalized

    def _normalize_state_type(self, state_type: str) -> str:
        if state_type not in STATE_TYPES:
            raise TraceLoggerError(f"invalid state_type {state_type!r}")
        return state_type

    def _normalize_owner_scope(self, owner_scope: str) -> str:
        if owner_scope not in OWNER_SCOPES:
            raise TraceLoggerError(f"invalid owner_scope {owner_scope!r}")
        return owner_scope

    def _normalize_tier(self, tier: str) -> str:
        if tier not in TIERS:
            raise TraceLoggerError(f"invalid tier {tier!r}")
        return tier

    def _positive_int(self, name: str, value: int) -> int:
        if value < 1:
            raise TraceLoggerError(f"{name} must be >= 1, got {value}")
        return int(value)

    def _non_negative_float(self, name: str, value: float) -> float:
        if value < 0:
            raise TraceLoggerError(f"{name} must be >= 0, got {value}")
        return float(value)

    def _base_payload(self, metadata: Mapping[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "workflow_id": self.workflow_id,
            "tenant_id": self.tenant_id,
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return payload

    def _next_ts(self) -> int:
        if self._clock is None:
            self._last_ts += 1
            return self._last_ts
        ts = int(self._clock())
        if ts < self._last_ts:
            raise TraceLoggerError(
                f"timestamps must be non-decreasing, got {ts} after {self._last_ts}"
            )
        self._last_ts = ts
        return ts

    def _write_event(self, op: str, payload: Mapping[str, object]) -> None:
        record = {"ts": self._next_ts(), "op": op, **payload}
        self._handle.write(json.dumps(record, sort_keys=False) + "\n")
        if self._auto_flush:
            self._handle.flush()
