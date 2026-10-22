"""Low-rate background fine-tuning from completed local Ollama exchanges."""

from __future__ import annotations

import json
import os
from collections import deque
import pathlib
import queue
import re
import threading
import time
import uuid
from typing import Optional

_SECRET_ASSIGNMENT = re.compile(
    r"(?im)\b(?:api[_-]?key|client[_-]?secret|secret|token|password|passwd|access[_-]?key)"
    r"\s*[:=]\s*['\"]?[^\s'\"]{6,}"
)
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE)
_SYSTEM_PROMPT = "You are CodeAI, a private coding and learning assistant."
_LOCK = threading.RLock()
_QUEUE: queue.Queue = queue.Queue()
_STARTED = False


def training_file() -> pathlib.Path:
    """Keep learned conversation data in the app's local data directory."""
    configured = os.environ.get("NEXUS_DATA_DIR")
    if configured:
        root = pathlib.Path(configured).expanduser()
    elif os.name == "nt":
        base = pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home() / "AppData" / "Local"))
        root = base / "Nexus"
    else:
        root = pathlib.Path.home() / ".local" / "share" / "nexus"
    return root / "training" / "ollama_conversations.jsonl"


def _safe_text(value: str, limit: int) -> str:
    text = str(value).strip()
    if not text or len(text) > limit or _SECRET_ASSIGNMENT.search(text) or _PRIVATE_KEY.search(text):
        return ""
    return text


def _append_json_line(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.name == "ollama_conversations.jsonl":
        _trim_corpus(path)


def _trim_corpus(path: pathlib.Path) -> None:
    try:
        if path.stat().st_size <= 20_000_000:
            return
        recent = deque(maxlen=1000)
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("question") and item.get("answer"):
                    recent.append(json.dumps(item, ensure_ascii=False))
        temporary = path.with_suffix(".trim.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for line in recent:
                stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        pass


def _load_pending(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("id") and item.get("question") and item.get("answer"):
                    records.append(item)
    except OSError:
        return []
    return records


def _remove_pending(path: pathlib.Path, item_id: str) -> None:
    pending = pathlib.Path(path)
    if not pending.name.endswith(".pending.jsonl"):
        pending = pending.with_suffix(".pending.jsonl")
    with _LOCK:
        records = [item for item in _load_pending(pending) if item.get("id") != item_id]
        temporary = pending.with_suffix(".pending.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for item in records:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pending)


def submit_example(question: str, answer: str, checkpoint: Optional[pathlib.Path] = None) -> bool:
    """Persist a safe pair and schedule one bounded model update in a daemon worker."""
    question = _safe_text(question, 4000)
    answer = _safe_text(answer, 16000)
    if not question or not answer:
        return False

    corpus = training_file()
    pending = corpus.with_suffix(".pending.jsonl")
    item = {"id": uuid.uuid4().hex, "question": question, "answer": answer,
            "created_at": time.time()}
    with _LOCK:
        _append_json_line(corpus, item)
        _ensure_worker_locked(corpus, pending, checkpoint)
        _append_json_line(pending, item)
        _QUEUE.put((item, checkpoint))
    return True


def _ensure_worker_locked(corpus: pathlib.Path, pending: pathlib.Path,
                          checkpoint: Optional[pathlib.Path]) -> None:
    global _STARTED
    if _STARTED:
        return
    for item in _load_pending(pending):
        _QUEUE.put((item, checkpoint))
    thread = threading.Thread(target=_worker, args=(pending,), name="nexus-live-learning", daemon=True)
    _STARTED = True
    thread.start()


def _worker(pending: pathlib.Path) -> None:
    while True:
        item, checkpoint = _QUEUE.get()
        try:
            _train_one(item, checkpoint)
        except Exception:
            # The durable pending record will be retried the next time the app starts.
            pass
        else:
            try:
                _remove_pending(pending, str(item["id"]))
            except OSError:
                # Keep the journal entry for retry if the pending file cannot be updated.
                pass
        finally:
            _QUEUE.task_done()


def _train_one(item: dict, checkpoint: Optional[pathlib.Path]) -> None:
    if checkpoint is None:
        configured = os.environ.get("CODEAI_PERSONAL_MODEL")
        checkpoint = pathlib.Path(configured).expanduser() if configured else (
            pathlib.Path(__file__).resolve().parent / "checkpoints" / "personal_char_lstm.npz")
    if not checkpoint.is_file():
        raise RuntimeError("The personal checkpoint is not available yet")

    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    from model.checkpoint_lock import checkpoint_lock
    from model import train as trainer
    from providers.brain_knowledge import BUILTIN_EXAMPLES
    from providers.personal import TRAINING_SYSTEM_PROMPT

    np = trainer.np
    rng = np.random.default_rng()
    records = [(f"System: {TRAINING_SYSTEM_PROMPT}\nUser: {item['question']}\nAssistant:",
                item["answer"])]
    examples = [example for example in BUILTIN_EXAMPLES
                if str(example.get("question", "")).strip() and str(example.get("answer", "")).strip()]
    if examples:
        selected = rng.choice(len(examples), size=min(2, len(examples)), replace=False)
        for index in np.atleast_1d(selected):
            example = examples[int(index)]
            records.append((f"System: {TRAINING_SYSTEM_PROMPT}\n"
                            f"User: {str(example['question']).strip()}\nAssistant:",
                            str(example["answer"]).strip()))

    with checkpoint_lock(checkpoint):
        params, moments, metadata = trainer.load_checkpoint(checkpoint)
        sequences, masks, _ = trainer.encode_records(records, answer_limit=384)
        selected_rows = np.arange(len(sequences), dtype=np.int64)
        x, y, loss_mask = trainer.make_batch(sequences, masks, selected_rows)
        loss, gradients = trainer.loss_and_gradients(params, x, y, loss_mask)
        norm = np.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients.values()))
        if norm > 1.0:
            gradients = {name: gradient / norm for name, gradient in gradients.items()}

        step = int(metadata.get("step", 0)) + 1
        beta1, beta2, epsilon, learning_rate = 0.9, 0.999, 1e-8, 0.00005
        for name, parameter in params.items():
            m, v = moments[f"m_{name}"], moments[f"v_{name}"]
            gradient = gradients[name]
            m *= beta1
            m += (1.0 - beta1) * gradient
            v *= beta2
            v += (1.0 - beta2) * gradient * gradient
            m_hat = m / (1.0 - beta1 ** step)
            v_hat = v / (1.0 - beta2 ** step)
            parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        metadata["step"] = step
        metadata["loss"] = float(loss)
        metadata["online_examples"] = int(metadata.get("online_examples", 0)) + 1
        training = metadata.setdefault("training", {})
        training["online_learning"] = "one low-rate update after each completed Ollama Ask turn or approved code edit"
        training["online_answer_bytes"] = 384
        trainer.save_checkpoint(checkpoint, params, moments, metadata)
