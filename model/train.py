"""Train CodeAI's small character LSTM from random weights with NumPy.

No pretrained weights are loaded. This is a compact CPU model, not a claim of
complete subject or language coverage. Add material you own or may use to
``model/training_data`` before training to expand its corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

try:
    from model.checkpoint_lock import checkpoint_lock
except ImportError:  # direct execution via python model/train.py
    from checkpoint_lock import checkpoint_lock


ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = pathlib.Path(__file__).resolve().parent
TRAINING_DIR = MODEL_DIR / "training_data"
CHECKPOINT = MODEL_DIR / "checkpoints" / "personal_char_lstm.npz"
EOS = 256
VOCAB_SIZE = 258
CONTEXT = 192
ANSWER_BYTES = 1536
EMBEDDING = 48
HIDDEN = 128
QA_REPEATS = 10


def build_records() -> list[tuple[str, str]]:
    """Build prompt/answer samples from local curated sources and user data."""
    sys.path.insert(0, str(ROOT))
    from providers.brain_knowledge import BUILTIN_EXAMPLES
    from providers.personal import TRAINING_SYSTEM_PROMPT

    qa_records: list[tuple[str, str]] = []
    source_records: list[tuple[str, str]] = []
    owner_records: list[tuple[str, str]] = []
    live_records: list[tuple[str, str]] = []
    system = TRAINING_SYSTEM_PROMPT
    for example in BUILTIN_EXAMPLES:
        for question in [example.get("question", ""), *example.get("aliases", [])]:
            question, answer = str(question).strip(), str(example.get("answer", "")).strip()
            if question and answer:
                qa_records.append((f"System: {system}\nUser: {question}\nAssistant:", answer))

    source_files = [
        "README.md", "providers/README.md", "providers/brain.py",
        "agent.py", "runner.py", "gui.py", "ui/index.html", "ui/site.html",
    ]
    for relative in source_files:
        path = ROOT / relative
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if not content.strip():
            continue
        data = content.encode("utf-8")
        width, overlap = 1800, 160
        for start in range(0, len(data), width - overlap):
            excerpt = data[start:start + width].decode("utf-8", errors="ignore")
            if len(excerpt.strip()) >= 80:
                question = f"Read this project reference excerpt from {relative}:"
                source_records.append((f"System: {system}\nUser: {question}\nAssistant:\n{excerpt}", ""))

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    allowed = {".txt", ".md", ".mdx", ".py", ".pyi", ".c", ".h", ".cpp", ".hpp",
               ".rs", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".java", ".go",
               ".cs", ".asm", ".s", ".html", ".css", ".scss", ".vue", ".svg", ".xml",
               ".json", ".jsonl", ".sql", ".rb", ".php", ".kt", ".swift", ".scala",
               ".dart", ".r", ".jl", ".ex", ".exs", ".clj", ".lua", ".pl", ".m",
               ".hs", ".ml", ".sol", ".graphql", ".glsl", ".vert", ".frag", ".wgsl",
               ".sh", ".zsh", ".ps1", ".psm1"}
    training_root = TRAINING_DIR.resolve()
    total_extra_bytes = 0
    for path in sorted(TRAINING_DIR.rglob("*")):
        name = path.name.casefold()
        relative_parts = path.relative_to(TRAINING_DIR).parts
        if (path.is_symlink() or not path.is_file() or path.suffix.lower() not in allowed
                or any(part.startswith(".") for part in relative_parts)
                or any(word in name for word in ("secret", "credential", "password", "private_key"))):
            continue
        try:
            if not path.resolve().is_relative_to(training_root):
                continue
            size = path.stat().st_size
            if size > 2_000_000 or total_extra_bytes + size > 50_000_000:
                continue
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if content.strip():
            total_extra_bytes += size
            label = path.relative_to(TRAINING_DIR).as_posix()
            owner_records.append((f"System: {system}\nUser: Study this added training document ({label}).\nAssistant:\n{content}", ""))

    # Import the local app-data journal lazily so the trainer can also run standalone.
    try:
        from model.live_learning import training_file
        live_path = training_file()
        if live_path.is_file():
            with live_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    question = str(item.get("question", "")).strip() if isinstance(item, dict) else ""
                    answer = str(item.get("answer", "")).strip() if isinstance(item, dict) else ""
                    if question and answer:
                        live_records.append((f"System: {system}\nUser: {question}\nAssistant:", answer))
    except OSError:
        pass

    # Give the small network more curated examples than raw app source.
    return qa_records * QA_REPEATS + source_records + owner_records + live_records


def encode_records(records: list[tuple[str, str]], answer_limit: int = ANSWER_BYTES):
    """Encode prompt/answer pairs; mask prompt bytes from the language loss."""
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    digest = hashlib.sha256()
    for prompt, answer in records:
        original = (prompt + answer + "\n").encode("utf-8")
        digest.update(original)
        if not answer:
            marker = "\nAssistant:"
            split = prompt.rfind(marker)
            if split < 0:
                continue
            answer = prompt[split + len(marker):]
            prompt = prompt[:split] + marker
        prefix = utf8_tail(prompt, CONTEXT)
        target = utf8_prefix(answer + "\n", answer_limit)
        if not prefix or not target:
            continue
        prefix_tokens = np.frombuffer(prefix, dtype=np.uint8).astype(np.uint16)
        target_tokens = np.concatenate((
            np.frombuffer(target, dtype=np.uint8).astype(np.uint16),
            np.array([EOS], dtype=np.uint16)))
        sequence = np.concatenate((prefix_tokens, target_tokens))
        loss_mask = np.zeros(len(sequence) - 1, dtype=np.float32)
        loss_mask[len(prefix_tokens) - 1:] = 1.0
        sequences.append(sequence)
        masks.append(loss_mask)
    if not sequences:
        raise RuntimeError("No training material was found.")
    return sequences, masks, digest.hexdigest()


def utf8_prefix(text: str, limit: int) -> bytes:
    raw = text.encode("utf-8")[:limit]
    while raw:
        try:
            raw.decode("utf-8")
            return raw
        except UnicodeDecodeError as exc:
            if exc.end == len(raw):
                raw = raw[:exc.start]
            else:
                return raw[:exc.start]
    return b""


def utf8_tail(text: str, limit: int) -> bytes:
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


def make_batch(sequences: list[np.ndarray], masks: list[np.ndarray],
               selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = len(selected)
    steps = max(len(sequences[int(index)]) - 1 for index in selected)
    x = np.zeros((batch, steps), dtype=np.uint16)
    y = np.zeros((batch, steps), dtype=np.uint16)
    loss_mask = np.zeros((batch, steps), dtype=np.float32)
    for row, index in enumerate(selected):
        sequence = sequences[int(index)]
        length = len(sequence) - 1
        x[row, :length] = sequence[:-1]
        y[row, :length] = sequence[1:]
        loss_mask[row, :length] = masks[int(index)]
    return x, y, loss_mask


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def initial_state(rng: np.random.Generator):
    params = {
        "embedding": (rng.standard_normal((VOCAB_SIZE, EMBEDDING)) * 0.04).astype(np.float32),
        "wx": (rng.standard_normal((EMBEDDING, HIDDEN * 4)) /
               np.sqrt(EMBEDDING)).astype(np.float32),
        "wh": (rng.standard_normal((HIDDEN, HIDDEN * 4)) /
               np.sqrt(HIDDEN)).astype(np.float32),
        "b": np.concatenate((np.zeros(HIDDEN, dtype=np.float32),
                             np.ones(HIDDEN, dtype=np.float32),
                             np.zeros(HIDDEN * 2, dtype=np.float32))),
        "wo": (rng.standard_normal((HIDDEN, VOCAB_SIZE)) /
               np.sqrt(HIDDEN)).astype(np.float32),
        "bo": np.zeros(VOCAB_SIZE, dtype=np.float32),
    }
    moments = {f"{kind}_{name}": np.zeros_like(value)
               for name, value in params.items() for kind in ("m", "v")}
    return params, moments


def loss_and_gradients(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray,
                       loss_mask: np.ndarray):
    batch, steps = x.shape
    hidden = np.zeros((batch, HIDDEN), dtype=np.float32)
    cell = np.zeros_like(hidden)
    caches = []
    token_ids = x.astype(np.int64)
    embedded = params["embedding"][token_ids]
    input_gates = (embedded.reshape(batch * steps, EMBEDDING) @ params["wx"]
                   + params["b"]).reshape(batch, steps, HIDDEN * 4)
    total_loss = 0.0
    rows = np.arange(batch)

    for t in range(steps):
        ids = token_ids[:, t]
        previous_hidden, previous_cell = hidden, cell
        gates = input_gates[:, t] + previous_hidden @ params["wh"]
        gi, gf, gg, go = np.split(gates, 4, axis=1)
        gate_i, gate_f, gate_g, gate_o = sigmoid(gi), sigmoid(gf), np.tanh(gg), sigmoid(go)
        cell = gate_f * previous_cell + gate_i * gate_g
        tanh_cell = np.tanh(cell)
        hidden = gate_o * tanh_cell
        logits = hidden @ params["wo"] + params["bo"]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        target = y[:, t].astype(np.int64)
        total_loss -= float(np.dot(np.log(np.maximum(probabilities[rows, target], 1e-9)),
                                   loss_mask[:, t]))
        caches.append((ids, previous_hidden, previous_cell, gate_i, gate_f,
                       gate_g, gate_o, tanh_cell, hidden, probabilities))

    gradients = {name: np.zeros_like(value) for name, value in params.items()}
    hidden_gradient = np.zeros_like(hidden)
    cell_gradient = np.zeros_like(cell)
    total_targets = max(1.0, float(loss_mask.sum()))
    gradient_cache = []
    for t in range(steps - 1, -1, -1):
        (ids, previous_hidden, previous_cell, gate_i, gate_f, gate_g,
         gate_o, tanh_cell, hidden_t, probabilities) = caches[t]
        dlogits = probabilities.copy()
        dlogits[rows, y[:, t].astype(np.int64)] -= 1.0
        dlogits *= (loss_mask[:, t:t + 1] / total_targets)

        dhidden = dlogits @ params["wo"].T + hidden_gradient
        dgate_o = dhidden * tanh_cell
        dcell = cell_gradient + dhidden * gate_o * (1.0 - tanh_cell * tanh_cell)
        dgate_f = dcell * previous_cell
        cell_gradient = dcell * gate_f
        dgate_i = dcell * gate_g
        dgate_g = dcell * gate_i
        dz = np.concatenate((dgate_i * gate_i * (1.0 - gate_i),
                             dgate_f * gate_f * (1.0 - gate_f),
                             dgate_g * (1.0 - gate_g * gate_g),
                             dgate_o * gate_o * (1.0 - gate_o)), axis=1)
        gradient_cache.append((ids, previous_hidden, hidden_t, dlogits, dz))
        hidden_gradient = dz @ params["wh"].T

    # Combine the per-time-step derivatives into a few larger matrix
    # operations. This is mathematically the same gradient and substantially
    # faster on a CPU than dozens of small BLAS calls per sequence.
    all_ids = np.concatenate([item[0] for item in gradient_cache])
    all_previous_hidden = np.concatenate([item[1] for item in gradient_cache], axis=0)
    all_hidden = np.concatenate([item[2] for item in gradient_cache], axis=0)
    all_dlogits = np.concatenate([item[3] for item in gradient_cache], axis=0)
    all_dz = np.concatenate([item[4] for item in gradient_cache], axis=0)
    gradients["wo"] = all_hidden.T @ all_dlogits
    gradients["bo"] = all_dlogits.sum(axis=0)
    gradients["wx"] = params["embedding"][all_ids].T @ all_dz
    gradients["wh"] = all_previous_hidden.T @ all_dz
    gradients["b"] = all_dz.sum(axis=0)
    np.add.at(gradients["embedding"], all_ids, all_dz @ params["wx"].T)
    return total_loss / total_targets, gradients


def save_checkpoint(path: pathlib.Path, params: dict[str, np.ndarray],
                    moments: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **params, **moments,
                            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    os.replace(temporary, path)


def load_checkpoint(path: pathlib.Path):
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata"].item()))
        expected = {"kind": "char-lstm", "context": CONTEXT,
                    "embedding": EMBEDDING, "hidden": HIDDEN, "vocab_size": VOCAB_SIZE}
        if metadata.get("architecture") != expected:
            raise RuntimeError("The saved model uses a different architecture.")
        if metadata.get("training", {}).get("objective") != "answer-only masked loss":
            raise RuntimeError("The saved model used the old training objective; start a fresh run without --resume.")
        names = ("embedding", "wx", "wh", "b", "wo", "bo")
        params = {name: source[name].copy() for name in names}
        moments = {f"{kind}_{name}": source[f"{kind}_{name}"].copy()
                   for name in names for kind in ("m", "v")}
    return params, moments, metadata


def _train_locked(steps: int, batch_size: int, learning_rate: float,
                  checkpoint_every: int, resume: bool, seed: int) -> None:
    records = build_records()
    sequences, masks, corpus_hash = encode_records(records)
    token_count = sum(len(sequence) for sequence in sequences)
    total_answer_targets = sum(float(mask.sum()) for mask in masks)
    if not sequences or total_answer_targets < 1:
        raise RuntimeError("No prompt-to-answer training examples were found.")

    rng = np.random.default_rng(seed)
    online_examples = 0
    if resume and CHECKPOINT.is_file():
        params, moments, metadata = load_checkpoint(CHECKPOINT)
        online_examples = int(metadata.get("online_examples", 0))
        start_step = int(metadata.get("step", 0))
        if metadata.get("corpus_hash") != corpus_hash:
            print("Training corpus changed since that checkpoint; continuing on the updated corpus.", flush=True)
        print(f"Resuming from step {start_step}; corpus has {len(records)} records, "
              f"{token_count:,} byte tokens, and {total_answer_targets:,.0f} answer targets.", flush=True)
    else:
        params, moments = initial_state(rng)
        start_step = 0
        print(f"Starting LSTM from random weights; corpus has {len(records)} prompt/answer records, "
              f"{token_count:,} byte tokens, and {total_answer_targets:,.0f} answer targets.", flush=True)

    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    began = time.time()
    processed_targets = 0
    for step in range(start_step + 1, start_step + steps + 1):
        selected = rng.integers(0, len(sequences), size=batch_size)
        x, y, loss_mask = make_batch(sequences, masks, selected)
        processed_targets += int(loss_mask.sum())
        loss, gradients = loss_and_gradients(params, x, y, loss_mask)
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
        if norm > 1.0:
            factor = 1.0 / norm
            gradients = {name: grad * factor for name, grad in gradients.items()}

        for name, parameter in params.items():
            m, v = moments[f"m_{name}"], moments[f"v_{name}"]
            gradient = gradients[name]
            m *= beta1
            m += (1.0 - beta1) * gradient
            v *= beta2
            v += (1.0 - beta2) * (gradient * gradient)
            m_hat = m / (1.0 - beta1 ** step)
            v_hat = v / (1.0 - beta2 ** step)
            parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        if step % 100 == 0 or step == start_step + steps:
            elapsed = max(time.time() - began, 1e-6)
            trained_steps = step - start_step
            throughput = processed_targets / elapsed
            print(f"step={step} answer_loss={loss:.4f} answer_bytes_per_sec={throughput:.0f}", flush=True)
        if step % checkpoint_every == 0 or step == start_step + steps:
            metadata = {
                "name": "CodeAI Personal Character LSTM",
                "version": 2,
                "step": step,
                "loss": loss,
                "corpus_hash": corpus_hash,
                "records": len(records),
                "tokens": int(token_count),
                "answer_targets": int(total_answer_targets),
                "online_examples": online_examples,
                "prompt_format": "System/User/Assistant UTF-8 byte sequence; loss is applied to answer and EOS only",
                "architecture": {"kind": "char-lstm", "context": CONTEXT,
                                 "embedding": EMBEDDING, "hidden": HIDDEN,
                                 "vocab_size": VOCAB_SIZE},
                "training": {"initialization": "random weights; no pretrained model",
                             "optimizer": "Adam", "learning_rate": learning_rate,
                             "qa_repeats": QA_REPEATS,
                             "answer_bytes": ANSWER_BYTES,
                             "objective": "answer-only masked loss"},
            }
            save_checkpoint(CHECKPOINT, params, moments, metadata)
            print(f"checkpoint={CHECKPOINT} step={step}", flush=True)
    print(f"Training finished in {time.time() - began:.1f}s. Current checkpoint: {CHECKPOINT}", flush=True)


def train(steps: int, batch_size: int, learning_rate: float,
          checkpoint_every: int, resume: bool, seed: int) -> None:
    with checkpoint_lock(CHECKPOINT):
        _train_locked(steps, batch_size, learning_rate, checkpoint_every, resume, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20000,
                        help="additional optimizer steps (default: 20000)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the personal LSTM checkpoint")
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("steps, batch size, and checkpoint interval must be positive")
    train(args.steps, args.batch_size, args.learning_rate,
          args.checkpoint_every, args.resume, args.seed)


if __name__ == "__main__":
    main()
