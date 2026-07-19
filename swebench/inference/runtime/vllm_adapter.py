from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol

from swebench.inference.runtime.segment_materializer import VLLMSegmentRequest


@dataclass(frozen=True)
class VLLMBackendRequest:
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    prompt_mode: str = "monolithic"
    segment_request: VLLMSegmentRequest | None = None
    extra_body: Dict[str, object] = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        if self.prompt_mode == "segment_aware" and self.segment_request is not None:
            return self.segment_request.to_openai_messages(
                fallback_system_prompt=self.system_prompt
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


class VLLMAdapter(Protocol):
    def complete(self, request: VLLMBackendRequest) -> str:
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
        client, mode = self._client()
        payload = {
            "model": request.model,
            "messages": request.to_messages(),
            "temperature": request.temperature,
        }
        if request.extra_body:
            payload["extra_body"] = request.extra_body
        if mode == "client":
            response = client.chat.completions.create(**payload)
            return response.choices[0].message.content or ""
        response = client.chat.completions.create(**payload)
        return response.choices[0].message.content or ""
