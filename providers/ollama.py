"""Local Ollama chat provider using Ollama's streaming HTTP API."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, List, Optional

from providers import ChatResult


class OllamaProvider:
    provider_id = "ollama"
    display_name = "Ollama (local)"
    default_model = "llama3.2"
    supports_edits = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.thinking_enabled = False
        self._validate_base_url(str(cfg.get("ollama_url", "http://127.0.0.1:11434")))

    def set_thinking(self, enabled: bool) -> None:
        self.thinking_enabled = bool(enabled)

    @property
    def base_url(self) -> str:
        return self._validate_base_url(str(self.cfg.get("ollama_url", "http://127.0.0.1:11434")))

    @staticmethod
    def _validate_base_url(value: str) -> str:
        parsed = urllib.parse.urlsplit(value.strip())
        hostname = (parsed.hostname or "").casefold()
        if (parsed.scheme != "http" or hostname not in ("localhost", "127.0.0.1", "::1")
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("Ollama must use a local URL such as http://127.0.0.1:11434")
        return value.strip().rstrip("/")

    def _url(self, endpoint: str) -> str:
        return self.base_url + endpoint

    def _get_json(self, endpoint: str) -> dict:
        request = urllib.request.Request(self._url(endpoint), headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read(2_000_000).decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Ollama is not reachable at {self.base_url}: {reason}") from None
        if not isinstance(data, dict):
            raise RuntimeError("Ollama returned an invalid response")
        return data

    def list_models(self) -> List[str]:
        data = self._get_json("/api/tags")
        models = data.get("models", [])
        if not isinstance(models, list):
            return []
        return sorted({str(item.get("name", "")).strip() for item in models
                       if isinstance(item, dict) and str(item.get("name", "")).strip()})

    def ensure_model(self) -> None:
        model = str(self.cfg.get("model", "")).strip()
        if not model:
            raise RuntimeError("Choose an Ollama model in Settings.")
        names = self.list_models()
        if model not in names and f"{model}:latest" not in names:
            raise RuntimeError(f"Ollama model '{model}' is not installed. Run 'ollama pull {model}' first.")

    def status(self) -> dict:
        model = str(self.cfg.get("model", "")).strip()
        try:
            names = self.list_models()
        except RuntimeError as exc:
            return {"provider_id": self.provider_id, "provider_name": self.display_name,
                    "ok": False, "models": [], "installed": False, "error": str(exc),
                    "supports_edits": self.supports_edits}
        installed = model in names or (":" not in model and f"{model}:latest" in names)
        return {"provider_id": self.provider_id, "provider_name": self.display_name,
                "ok": True, "models": names, "installed": installed,
                "error": "" if installed else f"Model '{model}' is not installed in Ollama.",
                "supports_edits": self.supports_edits, "from_scratch": False}

    def chat_stream(self, messages: List[dict], num_ctx: int, num_predict: int,
                    on_piece: Callable[[str], None],
                    on_thinking: Optional[Callable[[str], None]] = None,
                    on_thinking_done: Optional[Callable[[], None]] = None) -> ChatResult:
        model = str(self.cfg.get("model", "")).strip()
        model_key = model.casefold().split(":", 1)[0]
        temperature = max(0.0, min(2.0, float(self.cfg.get("temperature", 0.1))))
        output_limit = int(num_predict)
        if model_key == "qwen3":
            output_limit = max(output_limit, 1024)
        payload = {"model": model,
                   "messages": [{"role": str(m.get("role", "user")),
                                 "content": str(m.get("content", ""))}
                                for m in messages if m.get("role") in ("system", "user", "assistant")],
                   "stream": True,
                   "think": self.thinking_enabled,
                   "options": {"num_ctx": int(num_ctx), "num_predict": output_limit,
                               "temperature": temperature}}
        request = urllib.request.Request(
            self._url("/api/chat"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
            method="POST")
        chunks: list[str] = []
        thinking_chunks: list[str] = []
        content_buffer = ""
        inside_legacy_thinking = False
        # Qwen3 may place its thought stream in message.content even when the
        # separate Think toggle is off; keep this path live whenever it appears.
        qwen3_content_thinking = model_key == "qwen3"
        qwen3_prefix_checked = not qwen3_content_thinking
        qwen3_bare_mode = False
        qwen3_thought_parts: list[str] = []
        thought_stream_open = False
        # Other Ollama models may omit the opening marker, so hold their
        # ambiguous content until it can be classified without leaking it.
        hold_unmarked_content = self.thinking_enabled or model_key == "qwen3"
        content_classified = not hold_unmarked_content
        native_thinking_seen = False
        stats: dict = {"model": model}
        done_seen = False

        def answer_piece(piece: str) -> None:
            if piece:
                finish_thinking()
                chunks.append(piece)
                on_piece(piece)

        def thinking_piece(piece: str) -> None:
            nonlocal thought_stream_open
            if piece:
                thought_stream_open = True
                thinking_chunks.append(piece)
                if on_thinking:
                    on_thinking(piece)

        def finish_thinking() -> None:
            nonlocal thought_stream_open
            if thought_stream_open:
                thought_stream_open = False
                if on_thinking_done:
                    on_thinking_done()

        def parse_content(piece: str = "", *, final: bool = False) -> None:
            """Split Ollama thought markers, including a bare closing marker."""
            nonlocal content_buffer, inside_legacy_thinking
            nonlocal content_classified, qwen3_prefix_checked, qwen3_bare_mode
            content_buffer += piece
            open_tag, close_tag = "<think>", "</think>"
            if qwen3_content_thinking and not qwen3_prefix_checked:
                prefix = content_buffer.casefold()
                if (not final and len(prefix) < len(open_tag)
                        and open_tag.startswith(prefix)):
                    return
                if prefix.startswith(open_tag):
                    content_buffer = content_buffer[len(open_tag):]
                else:
                    qwen3_bare_mode = True
                qwen3_prefix_checked = True
                inside_legacy_thinking = True
            while content_buffer:
                if inside_legacy_thinking:
                    match = re.search(re.escape(close_tag), content_buffer, re.IGNORECASE)
                    if match:
                        safe = content_buffer[:match.start()]
                        if qwen3_bare_mode and safe:
                            qwen3_thought_parts.append(safe)
                        thinking_piece(safe)
                        content_buffer = content_buffer[match.end():]
                        inside_legacy_thinking = False
                        qwen3_bare_mode = False
                        content_classified = True
                        finish_thinking()
                        continue
                    keep = 0 if final else len(close_tag) - 1
                    if len(content_buffer) > keep:
                        end = len(content_buffer) - keep if keep else len(content_buffer)
                        safe = content_buffer[:end]
                        if qwen3_bare_mode and safe:
                            qwen3_thought_parts.append(safe)
                        thinking_piece(safe)
                        content_buffer = content_buffer[end:]
                    if final and qwen3_bare_mode and qwen3_thought_parts:
                        # If this Qwen response never closes its thought block,
                        # recover the generated text as the final answer.
                        fallback = "".join(qwen3_thought_parts)
                        qwen3_bare_mode = False
                        inside_legacy_thinking = False
                        content_classified = True
                        finish_thinking()
                        answer_piece(fallback)
                    return

                open_match = re.search(re.escape(open_tag), content_buffer, re.IGNORECASE)
                close_match = re.search(re.escape(close_tag), content_buffer, re.IGNORECASE)
                if open_match and (not close_match or open_match.start() < close_match.start()):
                    answer_piece(content_buffer[:open_match.start()])
                    content_buffer = content_buffer[open_match.end():]
                    inside_legacy_thinking = True
                    content_classified = True
                    continue
                if close_match:
                    # A few local models omit <think> but still terminate their
                    # reasoning with </think>. Treat the prefix as hidden thought.
                    thinking_piece(content_buffer[:close_match.start()])
                    content_buffer = content_buffer[close_match.end():]
                    content_classified = True
                    finish_thinking()
                    continue
                if not content_classified and not native_thinking_seen and not final:
                    # Keep an undecided prefix until it either contains a thought
                    # marker or the stream ends; otherwise reasoning leaks to UI.
                    return
                content_classified = True
                keep = 0 if final else len(open_tag) - 1
                if len(content_buffer) > keep:
                    end = len(content_buffer) - keep if keep else len(content_buffer)
                    answer_piece(content_buffer[:end])
                    content_buffer = content_buffer[end:]
                return

        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                for line in response:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line.decode("utf-8"))
                    except (UnicodeError, json.JSONDecodeError):
                        raise RuntimeError("Ollama returned an invalid streaming response") from None
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        raise RuntimeError(f"Ollama returned an error: {event['error']}")
                    message = event.get("message", {})
                    thought = str(message.get("thinking", "")) if isinstance(message, dict) else ""
                    if thought:
                        native_thinking_seen = True
                        if qwen3_content_thinking and not qwen3_prefix_checked and not content_buffer:
                            qwen3_prefix_checked = True
                            content_classified = True
                        thinking_piece(thought)
                    piece = str(message.get("content", "")) if isinstance(message, dict) else ""
                    if piece:
                        if native_thinking_seen:
                            finish_thinking()
                        parse_content(piece)
                    if event.get("done"):
                        parse_content(final=True)
                        finish_thinking()
                        done_seen = True
                        stats.update({key: event[key] for key in (
                            "model", "prompt_eval_count", "eval_count", "eval_duration",
                            "load_duration", "done_reason") if key in event})
        except KeyboardInterrupt:
            if qwen3_bare_mode:
                content_buffer = ""
                finish_thinking()
            elif content_classified or inside_legacy_thinking:
                parse_content(final=True)
            else:
                # An interrupted undecided prefix may be a thought without its
                # closing marker. Do not leak that partial text into the answer.
                content_buffer = ""
            return ChatResult("".join(chunks), stats, interrupted=True,
                              thinking="".join(thinking_chunks))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(2000).decode("utf-8", errors="replace")
            except OSError:
                detail = ""
            try:
                message = json.loads(detail).get("error", detail)
            except (json.JSONDecodeError, AttributeError):
                message = detail or exc.reason
            raise RuntimeError(f"Ollama request failed ({exc.code}): {message}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Ollama request failed: {reason}") from None
        if not done_seen:
            raise RuntimeError("Ollama ended the response before marking it complete")
        stats["finish_reason"] = stats.get("done_reason", "stop")
        return ChatResult("".join(chunks), stats, thinking="".join(thinking_chunks))
