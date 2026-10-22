# CodeAI Personal Model

This folder contains a small character-level recurrent neural language model trained from random weights. It
does not download, load, or fine-tune another model. The trainer uses NumPy and the local seed
corpus built by `train.py`.

## Train

From the CodeAI project folder:

```powershell
python -m pip install -r model/requirements.txt
python model/train.py --steps 20000
```

Training saves `checkpoints/personal_char_lstm.npz` every 500 steps. It reads each question as
context and applies the language loss only to the matching answer and end marker; curated Q&A
examples are repeated ten times more than raw project excerpts. Continue with:

```powershell
python model/train.py --resume --steps 20000
```

Place additional text or source examples under `training_data/`. The trainer accepts common
text, documentation, and source extensions, skips hidden and secret-named files, ignores
symlinks, and limits each input to 2 MB. Only use material you own or are allowed to use.

## Scope

The v2 LSTM predicts one UTF-8 byte at a time. It has a 192-byte prompt context, a 1,536-byte answer
training limit (about 384 ordinary text tokens), and about 136,000 parameters. Its seed corpus
consists of CodeAI's built-in reference examples and selected project files. High-confidence
matches use their saved local reference answer. Unreadable and incomplete generated drafts are
suppressed; other free-form replies can still be wrong. This small model is experimental and is not a substitute
for a large-scale general language model.

## Online learning from Ollama

Select Ollama in the Agent settings. The *Train NEXUS from Ollama chats* switch is off by default
to avoid extra CPU use. When enabled, completed Ask turns without workspace context and approved code edits that pass
checks are saved in local app data, then one bounded update runs in a background thread. Interrupted
or cut-off replies, file attachments, and project-context Ask requests are skipped. The saved examples
are included in a later full training run as well.

This does not make the small LSTM comparable to Ollama or a large language model. Ollama can be
wrong, and NEXUS can learn those errors. The live update is intentionally small; review important
answers and keep the switch off if the computer is getting hot.
