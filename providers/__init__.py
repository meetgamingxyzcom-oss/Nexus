"""Provider contract and registry for NEXUS.

Adapters translate a provider's API or local runtime into one streaming chat
interface. The core engine only consumes this contract, so adding a backend does
not require changing chat orchestration or the safe edit/review pipeline.

Provider adapters should read credentials from environment variables or the OS
credential store. Do not persist API keys in workspace configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Protocol


@dataclass
class ChatResult:
    """Normalized reply from any provider; stats use common token/finish keys."""

    text: str
    stats: Dict[str, object] = field(default_factory=dict)
    interrupted: bool = False
    thinking: str = ""


class ChatProvider(Protocol):
    provider_id: str
    display_name: str

    def list_models(self) -> List[str]: ...
    def ensure_model(self) -> None: ...
    def status(self) -> dict: ...
    def chat_stream(self, messages: List[dict], num_ctx: int, num_predict: int,
                    on_piece: Callable[[str], None]) -> ChatResult: ...


ProviderFactory = Callable[[dict], ChatProvider]
_REGISTRY: Dict[str, dict] = {}


def register_provider(provider_id: str, display_name: str, factory: ProviderFactory,
                      settings: tuple = ()) -> None:
    """Register a normalized provider adapter under a stable configuration id."""
    key = provider_id.strip().lower()
    if not key or key in _REGISTRY:
        raise ValueError(f"Invalid or duplicate provider id: {provider_id}")
    _REGISTRY[key] = {"id": key, "name": display_name, "factory": factory,
                      "settings": tuple(settings)}


def provider_catalog() -> List[dict]:
    """Return the provider choices the UI can display (never includes credentials)."""
    return [{"id": item["id"], "name": item["name"], "settings": list(item["settings"])}
            for item in _REGISTRY.values()]


def create_provider(cfg: dict) -> ChatProvider:
    provider_id = str(cfg.get("provider", "nexus-brain")).strip().lower()
    spec = _REGISTRY.get(provider_id)
    if spec is None:
        available = ", ".join(item["id"] for item in _REGISTRY.values()) or "none"
        raise ValueError(f"Unknown provider '{provider_id}'. Available providers: {available}")
    return spec["factory"](cfg)


from providers.brain import NexusBrainProvider  # noqa: E402  (register after contract definitions)
from providers.personal import PersonalModelProvider
from providers.ollama import OllamaProvider  # noqa: E402

register_provider("nexus-brain", "NEXUS Brain", NexusBrainProvider)
register_provider("codeai-personal", "CodeAI Personal Model", PersonalModelProvider)
register_provider("ollama", "Ollama (local)", OllamaProvider, settings=("ollama_url",))

__all__ = ["ChatProvider", "ChatResult", "NexusBrainProvider", "PersonalModelProvider", "OllamaProvider", "create_provider",
           "provider_catalog", "register_provider"]
