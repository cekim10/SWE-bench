from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Dict, Protocol, Tuple

from swebench.inference.runtime.segment_materializer import SegmentedGenerationRequest


@dataclass(frozen=True)
class VLLMBackendRequest:
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_tokens: int | None = None
    prompt_mode: str = "monolithic"
    segment_request: SegmentedGenerationRequest | None = None
    extra_body: Dict[str, object] = field(default_factory=dict)
    messages_override: Tuple[Tuple[str, str], ...] | None = None

    def to_messages(self) -> list[dict[str, str]]:
        if self.messages_override is not None:
            return [{"role": role, "content": content} for role, content in self.messages_override]
        if self.segment_request is not None:
            return self.segment_request.to_openai_messages(
                fallback_system_prompt=self.system_prompt
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

    def message_digest(self) -> str:
        return sha256(
            json.dumps(self.to_messages(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()


@dataclass(frozen=True)
class VLLMCompletionResult:
    text: str
    duration_ms: float
    message_digest: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    frontend_cache_hit: bool = False


class VLLMAdapter(Protocol):
    def complete(self, request: VLLMBackendRequest) -> str:
        ...

    def complete_with_details(self, request: VLLMBackendRequest) -> VLLMCompletionResult:
        ...


class OpenAICompatibleVLLMAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _client(self):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError("openai package is required for provider=vllm") from exc

        client_kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if hasattr(openai, "OpenAI"):
            return openai.OpenAI(**client_kwargs), "client"

        openai.base_url = self.base_url
        openai.api_key = self.api_key
        if hasattr(openai, "timeout"):
            openai.timeout = self.timeout
        return openai, "module"

    def complete(self, request: VLLMBackendRequest) -> str:
        return self.complete_with_details(request).text

    def complete_with_details(self, request: VLLMBackendRequest) -> VLLMCompletionResult:
        client, mode = self._client()
        payload = {
            "model": request.model,
            "messages": request.to_messages(),
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.extra_body:
            payload["extra_body"] = request.extra_body
        started_at = time.perf_counter()
        if mode == "client":
            response = client.chat.completions.create(**payload)
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            usage = getattr(response, "usage", None)
            return VLLMCompletionResult(
                text=response.choices[0].message.content or "",
                duration_ms=duration_ms,
                message_digest=request.message_digest(),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                frontend_cache_hit=bool(request.messages_override),
            )
        response = client.chat.completions.create(**payload)
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        usage = getattr(response, "usage", None)
        return VLLMCompletionResult(
            text=response.choices[0].message.content or "",
            duration_ms=duration_ms,
            message_digest=request.message_digest(),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            frontend_cache_hit=bool(request.messages_override),
        )
