from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Mapping, Sequence


class ResidencyTier(str, Enum):
    HBM = "HBM"
    CPU = "CPU"
    EVICTED = "EVICTED"


@dataclass(frozen=True)
class SegmentIdentity:
    logical_key: str
    module: str


@dataclass
class SegmentVersion:
    state_id: str
    identity: SegmentIdentity
    version: int
    size_bytes: int
    token_count: int
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


@dataclass(frozen=True)
class MaterializationGroup:
    group_id: str
    workflow_id: str
    consumer: str
    ordered_segment_ids: tuple[str, ...]
    prompt_id: str | None = None


@dataclass(frozen=True)
class ResidencySnapshot:
    segment_tiers: Dict[str, ResidencyTier]
    access_counts: Dict[str, int]
    hbm_bytes: int
    cpu_bytes: int


@dataclass(frozen=True)
class ResidencyDecision:
    pin_in_hbm: tuple[str, ...] = ()
    offload_to_cpu: tuple[str, ...] = ()
    evict: tuple[str, ...] = ()
    rematerialize_for_read: tuple[str, ...] = ()


@dataclass(frozen=True)
class VLLMSegmentRequest:
    ordered_segments: tuple[SegmentVersion, ...]
    enable_prefix_caching: bool = True

    def assembled_prompt(self, separator: str = "\n\n") -> str:
        return separator.join(
            segment.text for segment in self.ordered_segments if segment.text
        )

    def to_openai_messages(
        self,
        *,
        fallback_system_prompt: str | None = None,
        separator: str = "\n\n",
    ) -> list[dict[str, str]]:
        system_parts = []
        user_parts = []
        for segment in self.ordered_segments:
            if not segment.text:
                continue
            role = str(segment.metadata.get("role", "user"))
            if role == "system":
                system_parts.append(segment.text)
            else:
                user_parts.append(segment.text)

        messages: list[dict[str, str]] = []
        if system_parts:
            messages.append({"role": "system", "content": separator.join(system_parts)})
        elif fallback_system_prompt:
            messages.append({"role": "system", "content": fallback_system_prompt})
        messages.append({"role": "user", "content": separator.join(user_parts)})
        return messages


class WeakHeuristicSegmentPolicy:
    """Minimal online policy so the abstraction is evaluable without a predictor."""

    def __init__(self, *, ephemeral_offload_after_reads: int = 1) -> None:
        self.ephemeral_offload_after_reads = ephemeral_offload_after_reads

    def decide(
        self,
        *,
        materializer: "SegmentMaterializer",
        group: MaterializationGroup,
    ) -> ResidencyDecision:
        pin_in_hbm = []
        offload_to_cpu = []
        rematerialize_for_read = []
        active_ids = set(group.ordered_segment_ids)

        for state_id in group.ordered_segment_ids:
            segment = materializer.get_segment(state_id)
            tier = materializer.tier_for(state_id)
            if tier == ResidencyTier.EVICTED:
                rematerialize_for_read.append(state_id)
            if segment.is_shared or segment.is_immutable or not segment.is_ephemeral:
                pin_in_hbm.append(state_id)

        for state_id, segment in materializer.iter_segments():
            if state_id in active_ids:
                continue
            if materializer.tier_for(state_id) != ResidencyTier.HBM:
                continue
            if segment.is_ephemeral and materializer.access_count(state_id) >= self.ephemeral_offload_after_reads:
                offload_to_cpu.append(state_id)

        return ResidencyDecision(
            pin_in_hbm=tuple(dict.fromkeys(pin_in_hbm)),
            offload_to_cpu=tuple(dict.fromkeys(offload_to_cpu)),
            rematerialize_for_read=tuple(dict.fromkeys(rematerialize_for_read)),
        )


class SegmentMaterializer:
    """Runtime boundary for segment-granular materialization independent of model backend."""

    def __init__(self) -> None:
        self._segments: Dict[str, SegmentVersion] = {}
        self._tiers: Dict[str, ResidencyTier] = {}
        self._access_counts: Dict[str, int] = {}

    def register_segment(
        self,
        segment: SegmentVersion,
        *,
        initial_tier: ResidencyTier | None = None,
    ) -> None:
        self._segments[segment.state_id] = segment
        self._access_counts.setdefault(segment.state_id, 0)
        tier = initial_tier or segment.residency_hint
        if tier is None:
            tier = ResidencyTier.HBM if (segment.is_shared or segment.is_immutable) else ResidencyTier.CPU
        self._tiers[segment.state_id] = tier

    def release_segment(self, state_id: str) -> None:
        self._tiers[state_id] = ResidencyTier.EVICTED

    def build_group(
        self,
        *,
        group_id: str,
        workflow_id: str,
        consumer: str,
        ordered_segment_ids: Sequence[str],
        prompt_id: str | None = None,
    ) -> MaterializationGroup:
        return MaterializationGroup(
            group_id=group_id,
            workflow_id=workflow_id,
            consumer=consumer,
            ordered_segment_ids=tuple(ordered_segment_ids),
            prompt_id=prompt_id,
        )

    def get_segment(self, state_id: str) -> SegmentVersion:
        return self._segments[state_id]

    def iter_segments(self) -> Iterable[tuple[str, SegmentVersion]]:
        return self._segments.items()

    def tier_for(self, state_id: str) -> ResidencyTier:
        return self._tiers.get(state_id, ResidencyTier.EVICTED)

    def access_count(self, state_id: str) -> int:
        return self._access_counts.get(state_id, 0)

    def record_read(self, state_ids: Sequence[str]) -> None:
        for state_id in state_ids:
            self._access_counts[state_id] = self._access_counts.get(state_id, 0) + 1
            if self.tier_for(state_id) == ResidencyTier.EVICTED:
                self._tiers[state_id] = ResidencyTier.CPU

    def apply_decision(self, decision: ResidencyDecision) -> None:
        for state_id in decision.pin_in_hbm:
            self._tiers[state_id] = ResidencyTier.HBM
        for state_id in decision.offload_to_cpu:
            self._tiers[state_id] = ResidencyTier.CPU
        for state_id in decision.evict:
            self._tiers[state_id] = ResidencyTier.EVICTED
        for state_id in decision.rematerialize_for_read:
            if self.tier_for(state_id) == ResidencyTier.EVICTED:
                self._tiers[state_id] = ResidencyTier.CPU

    def prepare_group_read(
        self,
        group: MaterializationGroup,
        *,
        policy: WeakHeuristicSegmentPolicy | None = None,
    ) -> ResidencyDecision:
        policy = policy or WeakHeuristicSegmentPolicy()
        decision = policy.decide(materializer=self, group=group)
        self.apply_decision(decision)
        self.record_read(group.ordered_segment_ids)
        return decision

    def build_vllm_request(self, group: MaterializationGroup) -> VLLMSegmentRequest:
        return VLLMSegmentRequest(
            ordered_segments=tuple(
                self.get_segment(state_id) for state_id in group.ordered_segment_ids
            )
        )

    def snapshot(self) -> ResidencySnapshot:
        hbm_bytes = 0
        cpu_bytes = 0
        for state_id, segment in self._segments.items():
            tier = self.tier_for(state_id)
            if tier == ResidencyTier.HBM:
                hbm_bytes += segment.size_bytes
            elif tier == ResidencyTier.CPU:
                cpu_bytes += segment.size_bytes
        return ResidencySnapshot(
            segment_tiers=dict(self._tiers),
            access_counts=dict(self._access_counts),
            hbm_bytes=hbm_bytes,
            cpu_bytes=cpu_bytes,
        )
