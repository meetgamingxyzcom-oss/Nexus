"""CodeAI's private, from-scratch byte-level generative model provider."""

from __future__ import annotations

import codecs
import json
import os
import pathlib
import time
from typing import Callable, List, Optional

from providers import ChatResult
from providers.brain import NexusBrainProvider
from providers.brain_knowledge import BUILTIN_EXAMPLES

TRAINING_SYSTEM_PROMPT = "You are CodeAI, a private coding and learning assistant."


class PersonalModelProvider:
    provider_id = "codeai-personal"
    display_name = "CodeAI Personal Model"
    model_name = "codeai-personal-char-lstm-v1"
    supports_edits = True
    eos_token = 256
    bos_token = 257

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.thinking_enabled = False
        configured = os.environ.get("CODEAI_PERSONAL_MODEL")
        self.checkpoint = pathlib.Path(configured).expanduser() if configured else (
            pathlib.Path(__file__).resolve().parent.parent / "model" / "checkpoints" /
            "personal_char_lstm.npz")
        self._np = None
        self._state = None
        self._checkpoint_signature = None
        # Use bundled local references for high-confidence questions while the
        # from-scratch generator is still experimental. No external model is used.
        self._reference = NexusBrainProvider.__new__(NexusBrainProvider)
        self._reference.seed_examples = [
            dict(item, aliases=list(item.get("aliases", [])))
            for item in BUILTIN_EXAMPLES
        ]
        self._reference.examples = self._reference.seed_examples
        self._reference._user_examples = []
        self._reference._fit()

    def set_thinking(self, enabled: bool) -> None:
        self.thinking_enabled = bool(enabled)

    @staticmethod
    def _split_thinking(text: str) -> tuple[str, str]:
        """Separate a generated thought block from the answer.

        Some local checkpoints emit a bare closing ``</think>`` marker without
        the opening tag. Treat the text before that marker as thought content so
        it never leaks into the visible answer.
        """
        lowered = text.lower()
        close = lowered.find("</think>")
        if close < 0:
            return text, ""
        open_tag = lowered.rfind("<think>", 0, close)
        thought_start = open_tag + len("<think>") if open_tag >= 0 else 0
        thinking = text[thought_start:close].strip()
        answer_prefix = text[:open_tag] if open_tag >= 0 else ""
        answer_suffix = text[close + len("</think>"):].lstrip("\r\n \t")
        answer = answer_prefix + answer_suffix
        return answer, thinking
        self._reference_words = set()
        self._reference_bigrams = set()
        self._reference_trigrams = set()
        for example in BUILTIN_EXAMPLES:
            for text in [example.get("question", ""), *example.get("aliases", []),
                         example.get("answer", "")]:
                words = self._words(str(text))
                self._reference_words.update(words)
                self._reference_bigrams.update(zip(words, words[1:]))
                self._reference_trigrams.update(zip(words, words[1:], words[2:]))

    def _numpy(self):
        if self._np is None:
            try:
                import numpy as np
            except ImportError:
                raise RuntimeError("The personal model needs NumPy. Install the project's model requirements.") from None
            self._np = np
        return self._np

    def list_models(self) -> List[str]:
        return [self.model_name]

    def ensure_model(self) -> None:
        if str(self.cfg.get("model", self.model_name)) != self.model_name:
            raise RuntimeError(f"Choose the installed personal model '{self.model_name}'.")
        if not self.checkpoint.is_file():
            raise RuntimeError("No personal checkpoint is trained yet. Run model/train.py first.")
        self._load()

    def _load(self):
        try:
            stat = self.checkpoint.stat()
        except OSError:
            raise RuntimeError("No trained personal checkpoint found.") from None
        signature = (stat.st_mtime_ns, stat.st_size)
        if self._state is not None and signature == self._checkpoint_signature:
            return self._state
        np = self._numpy()
        try:
            with np.load(self.checkpoint, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"].item()))
                params = {name: saved[name].copy()
                          for name in ("embedding", "wx", "wh", "b", "wo", "bo")}
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"The personal model checkpoint could not be loaded: {exc}") from None
        architecture = metadata.get("architecture", {})
        context = int(architecture.get("context", 0))
        hidden = params["wh"].shape[0]
        embedding = params["embedding"].shape[1]
        if (architecture.get("kind") != "char-lstm" or context < 8 or
                params["embedding"].shape[0] != 258 or
                params["wx"].shape != (embedding, hidden * 4) or
                params["wh"].shape != (hidden, hidden * 4) or
                params["wo"].shape != (hidden, 258)):
            raise RuntimeError("The personal model checkpoint has an unsupported architecture.")
        self._state = {"params": params, "context": context, "metadata": metadata}
        self._checkpoint_signature = signature
        return self._state

    def status(self) -> dict:
        try:
            self._numpy()
            if not self.checkpoint.is_file():
                raise RuntimeError("No trained personal checkpoint found.")
            self._load()
            ok = True
        except RuntimeError as exc:
            error = str(exc)
            ok = False
        else:
            error = ""
        metadata = self._state["metadata"] if self._state else {}
        installed = bool(ok and str(self.cfg.get("model", self.model_name)) == self.model_name)
        return {"provider_id": self.provider_id, "provider_name": self.display_name,
                "ok": ok, "models": self.list_models(), "installed": installed, "error": error,
                "supports_edits": self.supports_edits,
                "training_step": metadata.get("step", 0),
                "training_loss": metadata.get("loss"),
                "online_examples": metadata.get("online_examples", 0),
                "from_scratch": True}

    def _step(self, token: int, hidden: object, cell: object, params=None):
        np = self._numpy()
        if params is None:
            params = self._load()["params"]
        embedded = params["embedding"][np.asarray([token], dtype=np.int64)]
        gates = embedded @ params["wx"] + hidden @ params["wh"] + params["b"]
        gi, gf, gg, go = np.split(gates, 4, axis=1)
        sigmoid = lambda value: 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
        gate_i, gate_f, gate_g, gate_o = sigmoid(gi), sigmoid(gf), np.tanh(gg), sigmoid(go)
        cell = gate_f * cell + gate_i * gate_g
        hidden = gate_o * np.tanh(cell)
        logits = (hidden @ params["wo"] + params["bo"])[0]
        return logits, hidden, cell

    @staticmethod
    def _last_user_prompt(messages: List[dict]) -> str:
        for message in reversed(messages):
            if str(message.get("role", "")).casefold() != "user":
                continue
            content = str(message.get("content", ""))
            if "\n\nTASK:\n" in content:
                content = content.rsplit("\n\nTASK:\n", 1)[-1]
            return content.strip()
        return ""

    @staticmethod
    def _words(text: str) -> list[str]:
        words: list[str] = []
        for piece in text.casefold().split():
            word = "".join(char for char in piece if char.isalpha())
            if len(word) >= 2:
                words.append(word)
        return words

    def _looks_readable(self, text: str) -> bool:
        words = self._words(text)
        if len(words) < 6:
            return False
        known = sum(word in self._reference_words for word in words)
        bigrams = list(zip(words, words[1:]))
        trigrams = list(zip(words, words[1:], words[2:]))
        known_bigrams = sum(pair in self._reference_bigrams for pair in bigrams)
        known_trigrams = sum(triple in self._reference_trigrams for triple in trigrams)
        return (known / len(words) >= 0.68
                and known_bigrams / max(1, len(bigrams)) >= 0.55
                and known_trigrams / max(1, len(trigrams)) >= 0.18)

    @staticmethod
    def _utf8_tail(text: str, limit: int) -> bytes:
        raw = text.encode("utf-8")[-limit:]
        while raw:
            try:
                raw.decode("utf-8")
                return raw
            except UnicodeDecodeError as exc:
                if exc.start == 0:
                    raw = raw[exc.end:]
                else:
                    raw = raw[:exc.start]
        return b""

    def _known_answer(self, question: str):
        if not question:
            return None
        answer = self._reference._solve_equation(question)
        if answer is None:
            answer = self._reference._safe_arithmetic(question)
        if answer is not None:
            return answer, 1.0, "local_math"
        example, score = self._reference._match(question)
        if example and score >= 0.78:
            return example["answer"], score, "local_reference"
        return None

    def chat_stream(self, messages: List[dict], num_ctx: int, num_predict: int,
                    on_piece: Callable[[str], None],
                    on_thinking: Optional[Callable[[str], None]] = None,
                    on_thinking_done: Optional[Callable[[], None]] = None) -> ChatResult:
        np = self._numpy()
        self.ensure_model()
        known = self._known_answer(self._last_user_prompt(messages))
        if known:
            answer, score, source = known
            visible = "Local reference answer:\n\n" + answer
            on_piece(visible)
            return ChatResult(visible,
                              {"model": self.model_name,
                               "prompt_tokens": 1,
                               "completion_tokens": max(1, (len(visible.encode("utf-8")) + 3) // 4),
                               "completion_duration_ns": 1,
                               "finish_reason": "stop",
                               "answer_source": source,
                               "reference_score": score})
        # Match the compact prompt format used during training. The app's long
        # agent system prompt otherwise displaces useful question context.
        prompt = (f"System: {TRAINING_SYSTEM_PROMPT}\n"
                  f"User: {self._last_user_prompt(messages)}\nAssistant:")
        prompt = prompt.encode("utf-8", errors="replace")
        generated: list[int] = []
        output_parts: list[str] = []
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        rng = np.random.default_rng()
        temperature = max(0.0, min(1.25, float(self.cfg.get("temperature", 0.7))))
        state = self._load()
        trained_answer_bytes = int(state["metadata"].get("training", {}).get(
            "answer_bytes", state["context"]))
        # num_predict is in approximate text tokens; this model emits bytes.
        max_bytes = max(4, min(int(num_predict) * 4, trained_answer_bytes))
        top_k = 16
        finish_reason = "length"
        began = time.perf_counter_ns()
        live_thinking_pending = ""
        live_thinking_opening_checked = False
        live_thinking_closed = False

        def stream_thinking_piece(piece: str) -> None:
            nonlocal live_thinking_pending, live_thinking_opening_checked, live_thinking_closed
            if not self.thinking_enabled or on_thinking is None or live_thinking_closed:
                return
            live_thinking_pending += piece
            open_tag, close_tag = "<think>", "</think>"
            if not live_thinking_opening_checked:
                lower_pending = live_thinking_pending.lower()
                if len(lower_pending) < len(open_tag) and open_tag.startswith(lower_pending):
                    return
                if lower_pending.startswith(open_tag):
                    live_thinking_pending = live_thinking_pending[len(open_tag):]
                live_thinking_opening_checked = True
            close_at = live_thinking_pending.lower().find(close_tag)
            if close_at >= 0:
                safe = live_thinking_pending[:close_at]
                if safe:
                    on_thinking(safe)
                live_thinking_pending = ""
                live_thinking_closed = True
                if on_thinking_done:
                    on_thinking_done()
                return
            keep = len(close_tag) - 1
            if len(live_thinking_pending) > keep:
                safe = live_thinking_pending[:-keep]
                live_thinking_pending = live_thinking_pending[-keep:]
                if safe:
                    on_thinking(safe)

        hidden = np.zeros((1, state["params"]["wh"].shape[0]), dtype=np.float32)
        cell = np.zeros_like(hidden)
        logits = np.zeros(258, dtype=np.float32)
        prompt_tail = self._utf8_tail(prompt.decode("utf-8", errors="replace"), state["context"])
        for token in prompt_tail:
            logits, hidden, cell = self._step(token, hidden, cell, state["params"])
        for _ in range(max_bytes):
            try:
                logits[self.bos_token] = -1e9
                if temperature <= 1e-6:
                    token = int(np.argmax(logits))
                else:
                    scaled = logits.astype(np.float64) / max(temperature, 0.05)
                    candidates = np.argpartition(scaled, -top_k)[-top_k:]
                    values = scaled[candidates]
                    probabilities = np.exp(values - values.max())
                    probabilities /= probabilities.sum()
                    token = int(rng.choice(candidates, p=probabilities))
                if token == self.eos_token:
                    finish_reason = "stop"
                    break
                if token > 255:
                    continue
                generated.append(token)
                logits, hidden, cell = self._step(token, hidden, cell, state["params"])
                piece = decoder.decode(bytes((token,)), final=False)
                if piece:
                    output_parts.append(piece)
                    stream_thinking_piece(piece)
            except KeyboardInterrupt:
                finish_reason = "interrupted"
                break
        tail = decoder.decode(b"", final=True)
        if tail:
            output_parts.append(tail)
            if finish_reason != "interrupted":
                stream_thinking_piece(tail)
        if (live_thinking_pending and self.thinking_enabled and on_thinking
                and not live_thinking_closed and finish_reason != "interrupted"):
            on_thinking(live_thinking_pending)
            live_thinking_pending = ""
        if finish_reason == "length" and max_bytes < int(num_predict) * 4:
            finish_reason = "model_limit"
        elapsed = max(1, time.perf_counter_ns() - began)
        raw_result_text = "".join(output_parts)
        result_text, thinking_text = self._split_thinking(raw_result_text)
        answer_source = "from_scratch_generation"
        ends_cleanly = result_text.rstrip().endswith((".", "!", "?", "`", "}", "]", ")"))
        # Validate the complete generation when it included an explicit thought
        # block. The hidden reasoning can carry the learned vocabulary and make
        # the shorter answer alone look unfamiliar to this small model's filter.
        readable_source = raw_result_text if thinking_text else result_text
        if not ends_cleanly or not self._looks_readable(readable_source):
            result_text = (
                "I don't have a reliable complete answer for this question yet. The "
                "from-scratch model produced an unreadable or incomplete draft. Try a "
                "question close to one of its "
                "bundled topics, add permitted material under model/training_data and "
                "retrain, or use NEXUS Brain for saved local Q&A references."
            )
            answer_source = "unreadable_draft_filtered"
            finish_reason = "stop"
        try:
            on_piece(result_text)
        except KeyboardInterrupt:
            finish_reason = "interrupted"
        return ChatResult(result_text,
                          {"model": self.model_name,
                           "prompt_tokens": max(1, (len(prompt) + 3) // 4),
                           "completion_tokens": max(1, (len(result_text.encode("utf-8")) + 3) // 4),
                           "completion_duration_ns": elapsed, "finish_reason": finish_reason,
                           "answer_source": answer_source},
                          interrupted=finish_reason == "interrupted", thinking=thinking_text)


__all__ = ["PersonalModelProvider"]
