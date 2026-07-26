from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol, Sequence


class SegmentRuntimeError(ValueError):
    """Base class for segment runtime domain errors."""


class UnknownSegmentError(SegmentRuntimeError):
    """Raised when a runtime operation references an unknown segment."""


class DuplicateSegmentError(SegmentRuntimeError):
    """Raised when a segment_id is registered twice."""


class InvalidVersionError(SegmentRuntimeError):
    """Raised when a new version does not advance semantic identity."""


class InvalidLookupError(SegmentRuntimeError):
    """Raised when a lookup targets a non-live or inconsistent segment."""


class ReclaimNotSupportedError(SegmentRuntimeError):
    """Raised when physical reclaim is delegated to the serving engine."""


class ResidencyTier(str, Enum):
    HBM = "HBM"
    CPU = "CPU"
    EVICTED = "EVICTED"


class ContextSegmentState(str, Enum):
    """Backward-compatible aggregate state view for existing callers."""

    UNREGISTERED = "UNREGISTERED"
    REGISTERED = "REGISTERED"
    RESIDENT = "RESIDENT"
    OFFLOADED = "OFFLOADED"
    INVALIDATED = "INVALIDATED"
    EVICTED = "EVICTED"
    RELEASED = "RELEASED"


class SegmentRole(str, Enum):
    MONOLITHIC = "monolithic"
    SYSTEM = "system"
    PLAN = "plan"
    EVIDENCE = "evidence"
    SCRATCH = "scratch"
    TASK = "task"
    SUMMARY = "summary"
    ARTIFACT = "artifact"
    REVIEW = "review"
    VERIFICATION = "verification"
    ROUTER = "router"
    USER = "user"


class SemanticState(str, Enum):
    REGISTERED = "REGISTERED"
    SUPERSEDED = "SUPERSEDED"
    RELEASED = "RELEASED"


class ResidencyState(str, Enum):
    UNMATERIALIZED = "UNMATERIALIZED"
    RESIDENT = "RESIDENT"
    EVICTED = "EVICTED"


class LookupStatus(str, Enum):
    HIT_RESIDENT = "HIT_RESIDENT"
    HIT_EVICTED = "HIT_EVICTED"
    MISS = "MISS"
    INVALID = "INVALID"


def _canonical_digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize_role(value: SegmentRole | str) -> SegmentRole:
    if isinstance(value, SegmentRole):
        return value
    normalized = str(value).strip().lower()
    alias_map = {
        "scratchpad": SegmentRole.SCRATCH,
        "tool_output": SegmentRole.EVIDENCE,
        "retrieved_document": SegmentRole.EVIDENCE,
        "reviewer": SegmentRole.REVIEW,
    }
    if normalized in alias_map:
        return alias_map[normalized]
    for role in SegmentRole:
        if role.value == normalized:
            return role
    raise SegmentRuntimeError(f"unknown segment role {value!r}")


@dataclass(frozen=True)
class SegmentIdentity:
    """Stable semantic identity for a context segment across versions."""

    logical_key: str
    module: str

    def __post_init__(self) -> None:
        if not self.logical_key:
            raise SegmentRuntimeError("logical_key must be non-empty")
        if not self.module:
            raise SegmentRuntimeError("module must be non-empty")


@dataclass(frozen=True)
class ExecutionContextKey:
    """Exact reuse key for one segment under one ordered prefix lineage."""

    segment_id: str
    predecessor_chain_hash: str
    model_id: str
    tokenizer_id: str
    inference_config_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "segment_id",
            "predecessor_chain_hash",
            "model_id",
            "tokenizer_id",
            "inference_config_hash",
        ):
            if not getattr(self, field_name):
                raise SegmentRuntimeError(f"{field_name} must be non-empty")

    @property
    def digest(self) -> str:
        return _canonical_digest(
            {
                "segment_id": self.segment_id,
                "predecessor_chain_hash": self.predecessor_chain_hash,
                "model_id": self.model_id,
                "tokenizer_id": self.tokenizer_id,
                "inference_config_hash": self.inference_config_hash,
            }
        )

    @classmethod
    def from_lineage(
        cls,
        *,
        segment_id: str,
        predecessor_segment_ids: Sequence[str],
        model_id: str,
        tokenizer_id: str,
        inference_config: Mapping[str, object],
    ) -> "ExecutionContextKey":
        predecessor_chain_hash = _canonical_digest(
            {
                "segment_id": segment_id,
                "predecessors": list(predecessor_segment_ids),
            }
        )
        inference_config_hash = _canonical_digest(dict(inference_config))
        return cls(
            segment_id=segment_id,
            predecessor_chain_hash=predecessor_chain_hash,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            inference_config_hash=inference_config_hash,
        )


@dataclass(frozen=True)
class ContextSegment:
    """Immutable semantic runtime object for independently materializable context."""

    state_id: str
    identity: SegmentIdentity
    version: int
    role: SegmentRole | str = SegmentRole.EVIDENCE
    content: tuple[int, ...] = ()
    predecessor_ids: tuple[str, ...] = ()
    size_bytes: int = 0
    token_count: int = 0
    workflow_id: str = "workflow"
    text: str = ""
    recompute_cost: float = 0.0
    reload_cost: float = 0.0
    supersedes: tuple[str, ...] = ()
    is_shared: bool = False
    is_immutable: bool = False
    is_ephemeral: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)
    residency_hint: ResidencyTier | None = None

    def __post_init__(self) -> None:
        if not self.state_id:
            raise SegmentRuntimeError("segment_id/state_id must be non-empty")
        if self.version < 1:
            raise SegmentRuntimeError("version must be >= 1")
        role = self.role
        if not isinstance(role, SegmentRole):
            metadata_role = self.metadata.get("role") if self.metadata else None
            resolved_role = role or metadata_role or self.identity.module
            role = _normalize_role(resolved_role)
            object.__setattr__(self, "role", role)
        if not self.content:
            fallback = tuple(self.text.encode("utf-8"))
            object.__setattr__(self, "content", fallback)
        if any(not isinstance(token, int) or token < 0 for token in self.content):
            raise SegmentRuntimeError("content must be a tuple of non-negative integers")
        if self.size_bytes < 0:
            raise SegmentRuntimeError("size_bytes must be >= 0")
        if self.token_count < 0:
            raise SegmentRuntimeError("token_count must be >= 0")
        if self.size_bytes == 0:
            object.__setattr__(self, "size_bytes", len(self.text.encode("utf-8")))
        if self.token_count == 0:
            object.__setattr__(self, "token_count", len(self.content))
        object.__setattr__(self, "predecessor_ids", tuple(self.predecessor_ids))
        object.__setattr__(self, "supersedes", tuple(self.supersedes))

    @property
    def segment_id(self) -> str:
        return self.state_id

    @property
    def logical_id(self) -> str:
        return self.identity.logical_key


@dataclass
class KVAssociation:
    """Opaque serving-engine association for one execution-context materialization."""

    execution_context_key: ExecutionContextKey
    handle: object
    size_bytes: int
    created_at: float
    last_accessed_at: float
    resident: bool = True
    tier: ResidencyTier = ResidencyTier.HBM
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class RuntimeRecord:
    semantic_state: SemanticState
    residency_state: ResidencyState
    associations: Dict[ExecutionContextKey, KVAssociation]
    created_at: float
    last_accessed_at: float | None
    released_at: float | None
    access_count: int
    last_lookup_status: LookupStatus | None = None
    pending_reclaim_at: float | None = None
    pending_reclaim_reason: str | None = None


@dataclass(frozen=True)
class RuntimeEvent:
    timestamp: float
    segment_id: str
    logical_id: str
    version: int
    operation: str
    from_semantic_state: str | None
    to_semantic_state: str | None
    from_residency_state: str | None
    to_residency_state: str | None
    lookup_status: str | None
    execution_context_digest: str | None
    reason: str | None
    size_bytes: int
    token_count: int
    request_id: str | None = None
    step_name: str | None = None
    iteration: int | None = None


@dataclass(frozen=True)
class LookupResult:
    status: LookupStatus
    association: KVAssociation | None
    observed: bool
    predicted: bool
    reason: str | None = None


@dataclass(frozen=True)
class ContextSegmentGroup:
    """Ordered segment set used for one generation step."""

    group_id: str
    workflow_id: str
    consumer: str
    ordered_segment_ids: tuple[str, ...]
    prompt_id: str | None = None
    request_segment_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeResidencySnapshot:
    """Observable placement state of the segment runtime at one step."""

    segment_tiers: Dict[str, ResidencyTier]
    segment_states: Dict[str, ContextSegmentState]
    access_counts: Dict[str, int]
    hbm_bytes: int
    cpu_bytes: int


@dataclass(frozen=True)
class RuntimePlacementDecision:
    """Online placement decision applied to a segment set."""

    pin_in_hbm: tuple[str, ...] = ()
    offload_to_cpu: tuple[str, ...] = ()
    evict: tuple[str, ...] = ()
    rematerialize_for_read: tuple[str, ...] = ()


@dataclass(frozen=True)
class SegmentedGenerationRequest:
    """Segment-aware request passed from the runtime layer to a serving adapter."""

    ordered_segments: tuple[ContextSegment, ...]
    request_segments: tuple[ContextSegment, ...] = ()
    fallback_system_prompt: str | None = None
    fallback_user_prompt: str | None = None
    enable_prefix_caching: bool = True

    @staticmethod
    def _approx_token_count(text: str) -> int:
        return max(1, (len(text) + 3) // 4) if text else 0

    @staticmethod
    def _role_label(segment: ContextSegment) -> str:
        role = segment.role.value if isinstance(segment.role, SegmentRole) else str(segment.role)
        return role.upper()

    @classmethod
    def _serialize_segment(cls, segment: ContextSegment) -> str:
        text = segment.text.strip()
        label = cls._role_label(segment)
        return f"[{label}]\n{text}" if text else f"[{label}]"

    def payload_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        prior_token_set: set[str] = set()
        seen_serialized_payloads: set[str] = set()
        cumulative_serialized_tokens = 0

        for position, segment in enumerate(self.request_segments or self.ordered_segments):
            serialized = self._serialize_segment(segment)
            serialized_tokens = self._approx_token_count(serialized)
            cumulative_serialized_tokens += serialized_tokens
            token_set = {token for token in serialized.lower().split() if token}
            overlap_tokens = len(token_set & prior_token_set)
            overlap_ratio = overlap_tokens / len(token_set) if token_set else 0.0
            exact_duplicate = serialized in seen_serialized_payloads
            rows.append(
                {
                    "position": position,
                    "state_id": segment.state_id,
                    "logical_id": segment.logical_id,
                    "module": segment.identity.module,
                    "role": self._role_label(segment).lower(),
                    "raw_token_count": segment.token_count,
                    "serialized_token_count": serialized_tokens,
                    "cumulative_serialized_tokens": cumulative_serialized_tokens,
                    "overlap_token_estimate": overlap_tokens,
                    "overlap_ratio": overlap_ratio,
                    "exact_duplicate": exact_duplicate,
                }
            )
            prior_token_set.update(token_set)
            seen_serialized_payloads.add(serialized)
        return rows

    def assembled_prompt(self, separator: str = "\n\n") -> str:
        system_prompt = self.fallback_system_prompt or ""
        user_prompt = self.fallback_user_prompt or ""
        extra_messages = self.to_openai_messages(separator=separator)
        assembled = []
        if extra_messages and extra_messages[0]["role"] == "system":
            system_prompt = extra_messages[0]["content"]
            extra_messages = extra_messages[1:]
        if extra_messages:
            user_prompt = extra_messages[-1]["content"]
        if system_prompt:
            assembled.append(system_prompt)
        if user_prompt:
            assembled.append(user_prompt)
        return separator.join(assembled)

    def to_openai_messages(
        self,
        *,
        fallback_system_prompt: str | None = None,
        separator: str = "\n\n",
    ) -> list[dict[str, str]]:
        system_parts = []
        user_parts = []
        base_system_prompt = fallback_system_prompt or self.fallback_system_prompt
        base_user_prompt = self.fallback_user_prompt
        for segment in self.request_segments or self.ordered_segments:
            if not segment.text:
                continue
            if segment.role == SegmentRole.SYSTEM:
                if base_system_prompt and segment.text.strip() == base_system_prompt.strip():
                    continue
                system_parts.append(self._serialize_segment(segment))
            else:
                user_parts.append(self._serialize_segment(segment))
        if base_user_prompt:
            user_parts.append(base_user_prompt)

        messages: list[dict[str, str]] = []
        if system_parts:
            if base_system_prompt:
                messages.append(
                    {"role": "system", "content": separator.join([base_system_prompt, *system_parts])}
                )
            else:
                messages.append({"role": "system", "content": separator.join(system_parts)})
        elif base_system_prompt:
            messages.append({"role": "system", "content": base_system_prompt})
        messages.append({"role": "user", "content": separator.join(user_parts)})
        return messages


class ServingEngineAdapter(Protocol):
    def materialize(
        self,
        segment: ContextSegment,
        execution_context: ExecutionContextKey,
    ) -> KVAssociation:
        ...

    def reclaim(self, association: KVAssociation) -> None:
        raise ReclaimNotSupportedError("physical reclaim is delegated to the serving engine")

    def available_memory_bytes(self) -> int | None:
        return None


class SyntheticMaterializationAdapter:
    """Deterministic test double used until a real engine-managed handle is available."""

    def __init__(self) -> None:
        self.materialize_calls: list[tuple[str, str]] = []

    def materialize(
        self,
        segment: ContextSegment,
        execution_context: ExecutionContextKey,
    ) -> KVAssociation:
        now = time.time()
        self.materialize_calls.append((segment.segment_id, execution_context.digest))
        return KVAssociation(
            execution_context_key=execution_context,
            handle={
                "opaque_handle": f"synthetic::{segment.segment_id}::{execution_context.digest[:16]}"
            },
            size_bytes=segment.size_bytes,
            created_at=now,
            last_accessed_at=now,
            resident=True,
            tier=ResidencyTier.HBM,
        )

    def reclaim(self, association: KVAssociation) -> None:
        raise ReclaimNotSupportedError(
            "synthetic adapter does not expose physical reclaim; treat as serving-engine delegated"
        )

    def available_memory_bytes(self) -> int | None:
        return None


class ReuseAwareRuntimePolicy:
    """Compatibility heuristic used until role-aware pressure handling is implemented."""

    def __init__(self, *, ephemeral_offload_after_reads: int = 1) -> None:
        self.ephemeral_offload_after_reads = ephemeral_offload_after_reads

    def decide(
        self,
        *,
        runtime: "SegmentRuntime" | None = None,
        materializer: "SegmentRuntime" | None = None,
        group: ContextSegmentGroup,
    ) -> RuntimePlacementDecision:
        runtime = runtime or materializer
        if runtime is None:
            raise ValueError("runtime or materializer is required")
        pin_in_hbm = []
        offload_to_cpu = []
        rematerialize_for_read = []
        active_ids = set(group.ordered_segment_ids)

        for state_id in group.ordered_segment_ids:
            if runtime.state_for(state_id) == ContextSegmentState.EVICTED:
                rematerialize_for_read.append(state_id)
            segment = runtime.get_segment(state_id)
            if segment.is_shared or segment.is_immutable or not segment.is_ephemeral:
                pin_in_hbm.append(state_id)

        for state_id, segment in runtime.iter_segments():
            if state_id in active_ids:
                continue
            if runtime.tier_for(state_id) != ResidencyTier.HBM:
                continue
            if segment.is_ephemeral and runtime.access_count(state_id) >= self.ephemeral_offload_after_reads:
                offload_to_cpu.append(state_id)

        return RuntimePlacementDecision(
            pin_in_hbm=tuple(dict.fromkeys(pin_in_hbm)),
            offload_to_cpu=tuple(dict.fromkeys(offload_to_cpu)),
            rematerialize_for_read=tuple(dict.fromkeys(rematerialize_for_read)),
        )


class SegmentRuntime:
    """Thin runtime layer between semantic workflow state and serving-engine KV reuse."""

    def __init__(
        self,
        *,
        serving_adapter: ServingEngineAdapter | None = None,
        clock: Callable[[], float] | None = None,
        reclaim_grace_by_role: Mapping[SegmentRole | str, float] | None = None,
        hbm_budget_bytes: int | None = None,
    ) -> None:
        self.serving_adapter = serving_adapter or SyntheticMaterializationAdapter()
        self._clock = clock or time.time
        self._segments: Dict[str, ContextSegment] = {}
        self._records: Dict[str, RuntimeRecord] = {}
        self._current_segment_id_by_logical_id: Dict[str, str] = {}
        self._events: list[RuntimeEvent] = []
        self._hbm_budget_bytes = hbm_budget_bytes
        self._active_request_metadata: Dict[str, object] | None = None
        self._reclaim_grace_by_role = {
            _normalize_role(role): float(grace)
            for role, grace in (reclaim_grace_by_role or {}).items()
        }

    def _now(self) -> float:
        return float(self._clock())

    def _record_event(
        self,
        *,
        segment: ContextSegment,
        operation: str,
        from_record: RuntimeRecord | None,
        to_record: RuntimeRecord | None,
        lookup_status: LookupStatus | None = None,
        execution_context: ExecutionContextKey | None = None,
        reason: str | None = None,
    ) -> None:
        self._events.append(
            RuntimeEvent(
                timestamp=self._now(),
                segment_id=segment.segment_id,
                logical_id=segment.logical_id,
                version=segment.version,
                operation=operation,
                from_semantic_state=(
                    from_record.semantic_state.value if from_record is not None else None
                ),
                to_semantic_state=(
                    to_record.semantic_state.value if to_record is not None else None
                ),
                from_residency_state=(
                    from_record.residency_state.value if from_record is not None else None
                ),
                to_residency_state=(
                    to_record.residency_state.value if to_record is not None else None
                ),
                lookup_status=lookup_status.value if lookup_status is not None else None,
                execution_context_digest=execution_context.digest if execution_context else None,
                reason=reason,
                size_bytes=segment.size_bytes,
                token_count=segment.token_count,
                request_id=(
                    str(self._active_request_metadata.get("request_id"))
                    if self._active_request_metadata is not None
                    and self._active_request_metadata.get("request_id") is not None
                    else None
                ),
                step_name=(
                    str(self._active_request_metadata.get("step_name"))
                    if self._active_request_metadata is not None
                    and self._active_request_metadata.get("step_name") is not None
                    else None
                ),
                iteration=(
                    int(self._active_request_metadata.get("iteration"))
                    if self._active_request_metadata is not None
                    and self._active_request_metadata.get("iteration") is not None
                    else None
                ),
            )
        )

    def events(self) -> list[RuntimeEvent]:
        return list(self._events)

    def get_segment(self, state_id: str) -> ContextSegment:
        if state_id not in self._segments:
            raise UnknownSegmentError(f"unknown segment {state_id!r}")
        return self._segments[state_id]

    def get_record(self, state_id: str) -> RuntimeRecord:
        if state_id not in self._records:
            raise UnknownSegmentError(f"unknown segment {state_id!r}")
        return self._records[state_id]

    def _derive_compat_state(self, state_id: str) -> ContextSegmentState:
        if state_id not in self._records:
            return ContextSegmentState.UNREGISTERED
        record = self._records[state_id]
        if record.semantic_state == SemanticState.RELEASED:
            return ContextSegmentState.RELEASED
        if record.semantic_state == SemanticState.SUPERSEDED:
            return ContextSegmentState.INVALIDATED
        if record.residency_state == ResidencyState.UNMATERIALIZED:
            return ContextSegmentState.REGISTERED
        if record.residency_state == ResidencyState.EVICTED:
            return ContextSegmentState.EVICTED
        active = [assoc for assoc in record.associations.values() if assoc.resident]
        if active and all(assoc.tier == ResidencyTier.CPU for assoc in active):
            return ContextSegmentState.OFFLOADED
        return ContextSegmentState.RESIDENT

    def state_for(self, state_id: str) -> ContextSegmentState:
        return self._derive_compat_state(state_id)

    def tier_for(self, state_id: str) -> ResidencyTier:
        if state_id not in self._records:
            return ResidencyTier.EVICTED
        record = self._records[state_id]
        active = [assoc for assoc in record.associations.values() if assoc.resident]
        if not active:
            return ResidencyTier.EVICTED
        if any(assoc.tier == ResidencyTier.HBM for assoc in active):
            return ResidencyTier.HBM
        return ResidencyTier.CPU

    def access_count(self, state_id: str) -> int:
        if state_id not in self._records:
            return 0
        return self._records[state_id].access_count

    def iter_segments(self) -> Iterable[tuple[str, ContextSegment]]:
        return self._segments.items()

    def current_segment_id_for(self, logical_id: str) -> str | None:
        return self._current_segment_id_by_logical_id.get(logical_id)

    def register(self, segment: ContextSegment) -> None:
        if segment.segment_id in self._segments:
            raise DuplicateSegmentError(f"segment {segment.segment_id!r} is already registered")
        current_segment_id = self._current_segment_id_by_logical_id.get(segment.logical_id)
        if current_segment_id is not None:
            current_segment = self._segments[current_segment_id]
            if segment.version <= current_segment.version:
                raise InvalidVersionError(
                    f"segment {segment.segment_id!r} must advance version for logical_id {segment.logical_id!r}"
                )
        now = self._now()
        self._segments[segment.segment_id] = segment
        self._records[segment.segment_id] = RuntimeRecord(
            semantic_state=SemanticState.REGISTERED,
            residency_state=ResidencyState.UNMATERIALIZED,
            associations={},
            created_at=now,
            last_accessed_at=None,
            released_at=None,
            access_count=0,
        )
        self._current_segment_id_by_logical_id[segment.logical_id] = segment.segment_id
        self._record_event(
            segment=segment,
            operation="REGISTER",
            from_record=None,
            to_record=self._records[segment.segment_id],
        )

    def register_segment(
        self,
        segment: ContextSegment,
        *,
        initial_tier: ResidencyTier | None = None,
    ) -> None:
        self.register(segment)
        if initial_tier == ResidencyTier.CPU:
            self.materialize_segment(segment.segment_id, tier=ResidencyTier.CPU)
        elif initial_tier == ResidencyTier.HBM:
            self.materialize_segment(segment.segment_id, tier=ResidencyTier.HBM)

    def _validate_live_segment_for_lookup(self, segment: ContextSegment) -> None:
        record = self.get_record(segment.segment_id)
        if record.semantic_state != SemanticState.REGISTERED:
            raise InvalidLookupError(
                f"segment {segment.segment_id!r} is not live: {record.semantic_state.value}"
            )
        if self._current_segment_id_by_logical_id.get(segment.logical_id) != segment.segment_id:
            raise InvalidLookupError(
                f"segment {segment.segment_id!r} is not the current version for logical_id {segment.logical_id!r}"
            )

    def _touch_record(
        self,
        *,
        record: RuntimeRecord,
        association: KVAssociation | None,
        status: LookupStatus,
    ) -> None:
        now = self._now()
        record.last_accessed_at = now
        record.access_count += 1
        record.last_lookup_status = status
        if association is not None:
            association.last_accessed_at = now

    def _materialize_for_lookup(
        self,
        *,
        segment: ContextSegment,
        record: RuntimeRecord,
        execution_context: ExecutionContextKey,
        original_status: LookupStatus,
        reason: str,
    ) -> KVAssociation:
        self.reclaim_expired()
        self._reclaim_for_pressure(
            required_bytes=max(segment.size_bytes, 0),
            exclude_state_ids={segment.segment_id},
        )
        previous_record = RuntimeRecord(**record.__dict__)
        association = self.serving_adapter.materialize(segment, execution_context)
        association.execution_context_key = execution_context
        association.resident = True
        if association.tier == ResidencyTier.EVICTED:
            association.tier = ResidencyTier.HBM
        record.associations[execution_context] = association
        record.residency_state = ResidencyState.RESIDENT
        self._touch_record(record=record, association=association, status=original_status)
        self._record_event(
            segment=segment,
            operation="MATERIALIZE_INTERNAL",
            from_record=previous_record,
            to_record=record,
            lookup_status=original_status,
            execution_context=execution_context,
            reason=reason,
        )
        return association

    def _segment_grace_seconds(self, segment: ContextSegment) -> float:
        return float(self._reclaim_grace_by_role.get(_normalize_role(segment.role), 0.0))

    def _schedule_or_reclaim(
        self,
        *,
        segment: ContextSegment,
        record: RuntimeRecord,
        reason: str,
    ) -> None:
        grace_seconds = self._segment_grace_seconds(segment)
        if grace_seconds <= 0:
            self._mark_associations_evicted(segment=segment, record=record, reason=reason)
            record.pending_reclaim_at = None
            record.pending_reclaim_reason = None
            return
        record.pending_reclaim_at = self._now() + grace_seconds
        record.pending_reclaim_reason = reason

    def reclaim_expired(self) -> int:
        reclaimed = 0
        now = self._now()
        for state_id, record in self._records.items():
            if record.pending_reclaim_at is None or record.pending_reclaim_at > now:
                continue
            segment = self._segments[state_id]
            self._mark_associations_evicted(
                segment=segment,
                record=record,
                reason=record.pending_reclaim_reason or "deferred_reclamation",
            )
            record.pending_reclaim_at = None
            record.pending_reclaim_reason = None
            reclaimed += 1
        return reclaimed

    def _resident_hbm_bytes(self) -> int:
        total = 0
        for state_id, segment in self._segments.items():
            record = self._records[state_id]
            if record.residency_state != ResidencyState.RESIDENT:
                continue
            if any(
                association.resident and association.tier == ResidencyTier.HBM
                for association in record.associations.values()
            ):
                total += segment.size_bytes
        return total

    def _pressure_victims(self, *, exclude_state_ids: set[str]) -> list[str]:
        role_priority = {
            SegmentRole.SCRATCH: 0,
            SegmentRole.EVIDENCE: 1,
            SegmentRole.PLAN: 2,
            SegmentRole.TASK: 3,
            SegmentRole.SYSTEM: 4,
        }
        candidates = []
        for state_id, segment in self._segments.items():
            if state_id in exclude_state_ids:
                continue
            record = self._records[state_id]
            if record.residency_state != ResidencyState.RESIDENT:
                continue
            if not any(
                association.resident and association.tier == ResidencyTier.HBM
                for association in record.associations.values()
            ):
                continue
            semantic_rank = 0 if record.semantic_state != SemanticState.REGISTERED else 1
            candidates.append(
                (
                    semantic_rank,
                    role_priority.get(_normalize_role(segment.role), 5),
                    record.last_accessed_at if record.last_accessed_at is not None else float("-inf"),
                    -segment.size_bytes,
                    state_id,
                )
            )
        candidates.sort()
        return [state_id for _, _, _, _, state_id in candidates]

    def _reclaim_for_pressure(
        self,
        *,
        required_bytes: int,
        exclude_state_ids: set[str],
    ) -> int:
        if self._hbm_budget_bytes is None or required_bytes <= 0:
            return 0
        reclaimed_bytes = 0
        self.reclaim_expired()
        while self._resident_hbm_bytes() + required_bytes > self._hbm_budget_bytes:
            victims = self._pressure_victims(exclude_state_ids=exclude_state_ids)
            if not victims:
                break
            victim_id = victims[0]
            record = self._records[victim_id]
            segment = self._segments[victim_id]
            reclaimed_bytes += segment.size_bytes
            self._mark_associations_evicted(
                segment=segment,
                record=record,
                reason="policy_eviction",
            )
            record.pending_reclaim_at = None
            record.pending_reclaim_reason = None
        return reclaimed_bytes

    def lookup(
        self,
        segment: str | ContextSegment,
        execution_context: ExecutionContextKey | None = None,
    ) -> LookupResult | bool:
        if execution_context is None:
            state_id = segment.segment_id if isinstance(segment, ContextSegment) else str(segment)
            return self.state_for(state_id) == ContextSegmentState.RESIDENT

        resolved_segment = self.get_segment(segment) if isinstance(segment, str) else segment
        record = self.get_record(resolved_segment.segment_id)
        previous_record = RuntimeRecord(**record.__dict__)
        try:
            self._validate_live_segment_for_lookup(resolved_segment)
        except InvalidLookupError as exc:
            record.last_lookup_status = LookupStatus.INVALID
            self._record_event(
                segment=resolved_segment,
                operation="INVALID_LOOKUP",
                from_record=previous_record,
                to_record=record,
                lookup_status=LookupStatus.INVALID,
                execution_context=execution_context,
                reason=str(exc),
            )
            return LookupResult(
                status=LookupStatus.INVALID,
                association=None,
                observed=True,
                predicted=False,
                reason=str(exc),
            )

        association = record.associations.get(execution_context)
        if association is None:
            association = self._materialize_for_lookup(
                segment=resolved_segment,
                record=record,
                execution_context=execution_context,
                original_status=LookupStatus.MISS,
                reason="first_materialization",
            )
            return LookupResult(
                status=LookupStatus.MISS,
                association=association,
                observed=True,
                predicted=False,
                reason="first_materialization",
            )

        if association.resident:
            self._touch_record(
                record=record,
                association=association,
                status=LookupStatus.HIT_RESIDENT,
            )
            self._record_event(
                segment=resolved_segment,
                operation="REUSE",
                from_record=previous_record,
                to_record=record,
                lookup_status=LookupStatus.HIT_RESIDENT,
                execution_context=execution_context,
                reason="exact_context_match",
            )
            return LookupResult(
                status=LookupStatus.HIT_RESIDENT,
                association=association,
                observed=True,
                predicted=True,
                reason="exact_context_match",
            )

        association = self._materialize_for_lookup(
            segment=resolved_segment,
            record=record,
            execution_context=execution_context,
            original_status=LookupStatus.HIT_EVICTED,
            reason="rematerialization_after_eviction",
        )
        return LookupResult(
            status=LookupStatus.HIT_EVICTED,
            association=association,
            observed=True,
            predicted=False,
            reason="rematerialization_after_eviction",
        )

    def _mark_associations_evicted(
        self,
        *,
        segment: ContextSegment,
        record: RuntimeRecord,
        reason: str,
    ) -> None:
        if not record.associations:
            record.residency_state = ResidencyState.EVICTED
            return
        for association in record.associations.values():
            if association.resident:
                try:
                    self.serving_adapter.reclaim(association)
                except ReclaimNotSupportedError:
                    pass
                association.resident = False
                association.tier = ResidencyTier.EVICTED
                self._record_event(
                    segment=segment,
                    operation="RECLAIM",
                    from_record=record,
                    to_record=record,
                    execution_context=association.execution_context_key,
                    reason=reason,
                )
        record.residency_state = ResidencyState.EVICTED

    def supersede(self, old_segment: str | ContextSegment, new_segment: ContextSegment) -> None:
        old = self.get_segment(old_segment) if isinstance(old_segment, str) else old_segment
        old_record = self.get_record(old.segment_id)
        if old_record.semantic_state != SemanticState.REGISTERED:
            raise InvalidVersionError(
                f"old segment {old.segment_id!r} must be REGISTERED before supersede"
            )
        if old.logical_id != new_segment.logical_id:
            raise InvalidVersionError("supersede requires identical logical_id")
        if new_segment.version <= old.version:
            raise InvalidVersionError("superseding segment must increase version")
        self.register(new_segment)
        old_previous = RuntimeRecord(**old_record.__dict__)
        old_record.semantic_state = SemanticState.SUPERSEDED
        self._current_segment_id_by_logical_id[old.logical_id] = new_segment.segment_id
        self._schedule_or_reclaim(
            segment=old,
            record=old_record,
            reason="supersede_reclamation",
        )
        self._record_event(
            segment=old,
            operation="SUPERSEDE",
            from_record=old_previous,
            to_record=old_record,
            reason=f"superseded_by:{new_segment.segment_id}",
        )

    def supersede_segment(
        self,
        old_state_id: str,
        new_segment: ContextSegment,
        *,
        initial_tier: ResidencyTier | None = None,
    ) -> None:
        self.supersede(old_state_id, new_segment)
        if initial_tier == ResidencyTier.CPU:
            self.materialize_segment(new_segment.segment_id, tier=ResidencyTier.CPU)
        elif initial_tier == ResidencyTier.HBM:
            self.materialize_segment(new_segment.segment_id, tier=ResidencyTier.HBM)

    def release(self, segment: str | ContextSegment) -> None:
        resolved_segment = self.get_segment(segment) if isinstance(segment, str) else segment
        record = self.get_record(resolved_segment.segment_id)
        previous = RuntimeRecord(**record.__dict__)
        if record.semantic_state == SemanticState.RELEASED:
            return
        record.semantic_state = SemanticState.RELEASED
        record.released_at = self._now()
        self._schedule_or_reclaim(
            segment=resolved_segment,
            record=record,
            reason="release_reclamation",
        )
        if self._current_segment_id_by_logical_id.get(resolved_segment.logical_id) == resolved_segment.segment_id:
            del self._current_segment_id_by_logical_id[resolved_segment.logical_id]
        self._record_event(
            segment=resolved_segment,
            operation="RELEASE",
            from_record=previous,
            to_record=record,
            reason="semantic_release",
        )

    def release_segment(self, state_id: str) -> None:
        self.release(state_id)

    def materialize_segment(
        self,
        state_id: str,
        *,
        tier: ResidencyTier = ResidencyTier.HBM,
    ) -> None:
        segment = self.get_segment(state_id)
        record = self.get_record(state_id)
        if record.semantic_state == SemanticState.RELEASED:
            raise InvalidLookupError(f"cannot materialize released segment {state_id!r}")
        execution_context = ExecutionContextKey.from_lineage(
            segment_id=segment.segment_id,
            predecessor_segment_ids=segment.predecessor_ids,
            model_id="synthetic",
            tokenizer_id="synthetic",
            inference_config={"tier": tier.value},
        )
        association = self.serving_adapter.materialize(segment, execution_context)
        association.tier = tier if tier != ResidencyTier.EVICTED else ResidencyTier.HBM
        association.resident = True
        record.associations[execution_context] = association
        record.residency_state = ResidencyState.RESIDENT
        record.last_lookup_status = LookupStatus.MISS
        self._record_event(
            segment=segment,
            operation="MATERIALIZE_INTERNAL",
            from_record=record,
            to_record=record,
            lookup_status=LookupStatus.MISS,
            execution_context=execution_context,
            reason="compat_materialize_segment",
        )

    def offload_segment(self, state_id: str) -> None:
        record = self.get_record(state_id)
        if record.residency_state != ResidencyState.RESIDENT:
            raise InvalidLookupError(
                f"offload requires RESIDENT segment, got {record.residency_state.value}"
            )
        for association in record.associations.values():
            if association.resident:
                association.tier = ResidencyTier.CPU

    def invalidate_segment(self, state_id: str) -> None:
        record = self.get_record(state_id)
        if record.semantic_state == SemanticState.RELEASED:
            raise InvalidLookupError(f"cannot invalidate released segment {state_id!r}")
        self._mark_associations_evicted(
            segment=self.get_segment(state_id),
            record=record,
            reason="compat_invalidate",
        )
        record.semantic_state = SemanticState.SUPERSEDED

    def evict_segment(self, state_id: str) -> None:
        record = self.get_record(state_id)
        if record.semantic_state == SemanticState.RELEASED:
            raise InvalidLookupError(f"cannot evict released segment {state_id!r}")
        self._mark_associations_evicted(
            segment=self.get_segment(state_id),
            record=record,
            reason="policy_eviction",
        )

    def build_group(
        self,
        *,
        group_id: str,
        workflow_id: str,
        consumer: str,
        ordered_segment_ids: Sequence[str],
        prompt_id: str | None = None,
        request_segment_ids: Sequence[str] | None = None,
    ) -> ContextSegmentGroup:
        return ContextSegmentGroup(
            group_id=group_id,
            workflow_id=workflow_id,
            consumer=consumer,
            ordered_segment_ids=tuple(ordered_segment_ids),
            prompt_id=prompt_id,
            request_segment_ids=tuple(request_segment_ids or ordered_segment_ids),
        )

    def execution_context_for_group(
        self,
        *,
        group: ContextSegmentGroup,
        state_id: str,
        model_id: str = "group-model",
        tokenizer_id: str = "group-tokenizer",
        inference_config: Mapping[str, object] | None = None,
    ) -> ExecutionContextKey:
        ordered_ids = list(group.ordered_segment_ids)
        try:
            index = ordered_ids.index(state_id)
        except ValueError as exc:
            raise UnknownSegmentError(
                f"state_id {state_id!r} is not part of group {group.group_id!r}"
            ) from exc
        return ExecutionContextKey.from_lineage(
            segment_id=state_id,
            predecessor_segment_ids=ordered_ids[:index],
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            inference_config=dict(inference_config or {"consumer": group.consumer}),
        )

    def record_read(self, state_ids: Sequence[str]) -> None:
        for state_id in state_ids:
            record = self.get_record(state_id)
            record.access_count += 1
            record.last_accessed_at = self._now()

    def apply_decision(self, decision: RuntimePlacementDecision) -> None:
        for state_id in decision.pin_in_hbm:
            if self.state_for(state_id) != ContextSegmentState.RESIDENT:
                self.materialize_segment(state_id, tier=ResidencyTier.HBM)
        for state_id in decision.offload_to_cpu:
            if self.state_for(state_id) == ContextSegmentState.RESIDENT:
                self.offload_segment(state_id)
        for state_id in decision.evict:
            if self.state_for(state_id) != ContextSegmentState.RELEASED:
                self.evict_segment(state_id)
        for state_id in decision.rematerialize_for_read:
            if self.state_for(state_id) != ContextSegmentState.RESIDENT:
                self.materialize_segment(state_id, tier=ResidencyTier.HBM)

    def prepare_group_read(
        self,
        group: ContextSegmentGroup,
        *,
        policy: ReuseAwareRuntimePolicy | None = None,
        model_id: str = "group-model",
        tokenizer_id: str = "group-tokenizer",
        inference_config: Mapping[str, object] | None = None,
        request_metadata: Mapping[str, object] | None = None,
    ) -> RuntimePlacementDecision:
        self.reclaim_expired()
        policy = policy or ReuseAwareRuntimePolicy()
        previous_request_metadata = self._active_request_metadata
        self._active_request_metadata = dict(request_metadata or {})
        try:
            decision = policy.decide(runtime=self, group=group)
            self.apply_decision(decision)
            for state_id in group.ordered_segment_ids:
                lookup_result = self.lookup(
                    state_id,
                    self.execution_context_for_group(
                        group=group,
                        state_id=state_id,
                        model_id=model_id,
                        tokenizer_id=tokenizer_id,
                        inference_config=inference_config,
                    ),
                )
                if (
                    isinstance(lookup_result, LookupResult)
                    and lookup_result.status == LookupStatus.INVALID
                ):
                    raise InvalidLookupError(
                        f"invalid grouped lookup for {state_id!r}: {lookup_result.reason}"
                    )
        finally:
            self._active_request_metadata = previous_request_metadata
        return decision

    def build_vllm_request(
        self,
        group: ContextSegmentGroup,
        *,
        fallback_system_prompt: str | None = None,
        fallback_user_prompt: str | None = None,
    ) -> SegmentedGenerationRequest:
        return SegmentedGenerationRequest(
            ordered_segments=tuple(self.get_segment(state_id) for state_id in group.ordered_segment_ids),
            request_segments=tuple(
                self.get_segment(state_id) for state_id in group.request_segment_ids
            ),
            fallback_system_prompt=fallback_system_prompt,
            fallback_user_prompt=fallback_user_prompt,
        )

    def snapshot(self) -> RuntimeResidencySnapshot:
        segment_tiers: Dict[str, ResidencyTier] = {}
        segment_states: Dict[str, ContextSegmentState] = {}
        access_counts: Dict[str, int] = {}
        hbm_bytes = 0
        cpu_bytes = 0
        for state_id, segment in self._segments.items():
            segment_tiers[state_id] = self.tier_for(state_id)
            segment_states[state_id] = self.state_for(state_id)
            access_counts[state_id] = self.access_count(state_id)
            if segment_tiers[state_id] == ResidencyTier.HBM and segment_states[state_id] == ContextSegmentState.RESIDENT:
                hbm_bytes += segment.size_bytes
            elif segment_tiers[state_id] == ResidencyTier.CPU and segment_states[state_id] == ContextSegmentState.OFFLOADED:
                cpu_bytes += segment.size_bytes
        return RuntimeResidencySnapshot(
            segment_tiers=segment_tiers,
            segment_states=segment_states,
            access_counts=access_counts,
            hbm_bytes=hbm_bytes,
            cpu_bytes=cpu_bytes,
        )


# Backward-compatible aliases for earlier prototype names.
SegmentVersion = ContextSegment
MaterializationGroup = ContextSegmentGroup
ResidencySnapshot = RuntimeResidencySnapshot
ResidencyDecision = RuntimePlacementDecision
VLLMSegmentRequest = SegmentedGenerationRequest
WeakHeuristicSegmentPolicy = ReuseAwareRuntimePolicy
SegmentMaterializer = SegmentRuntime
SegmentRegistry = SegmentRuntime
