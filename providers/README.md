# NEXUS Brain and provider adapters

Ollama (local) is the default generative provider. It streams from the local Ollama HTTP
server and uses models already installed on this computer. Set the local server address under
Settings → Ollama (local) connection. No provider credentials are stored.

NEXUS Brain Lite matches questions to a curated local reference library and owner-supplied
examples, and evaluates a bounded set of arithmetic operations directly. Owner answers are kept in the app's private data directory. It has no pretrained
model, service dependency or code-editing ability. See the main README for its limits and
the Settings → Brain training workflow.

CodeAI Personal Model is a separate, from-scratch character-level recurrent neural language model. It runs from
the checkpoint in `model/checkpoints/personal_char_lstm.npz`, needs NumPy, and does not download or
load pretrained weights. See the main README and `model/train.py` to build and continue training it.

The provider package isolates model runtimes from the chat engine and the
web UI. Keep the adapter small: the engine owns workspace context, chat history, modes,
diff generation, review, backups, syntax checks and undo. The adapter owns connection
details, model discovery, streaming and translating the service response into NEXUS's
normalized result.

## Adapter contract

Implement the `ChatProvider` protocol in `providers/__init__.py`:

```python
class ExampleProvider:
    provider_id = "example"
    display_name = "Example"

    def __init__(self, cfg: dict): ...
    def list_models(self) -> list[str]: ...
    def ensure_model(self) -> None: ...
    def status(self) -> dict: ...
    def chat_stream(self, messages, num_ctx, num_predict, on_piece) -> ChatResult: ...
```

- `list_models` returns the model names the provider can use.
- `ensure_model` confirms that the configured model is usable, or raises `RuntimeError`
  with a concise message that can be shown to the user. It can be a no-op for providers
  where model names are not discoverable.
- `status` returns `provider_id`, `provider_name`, `ok`, `models`, `installed` and `error`.
  `error` should be empty when the provider is reachable.
- `chat_stream` receives the engine's OpenAI-style `[{"role", "content"}]` messages,
  the configured context/output limits, and `on_piece` to forward each text chunk. Return
  `ChatResult(full_text, normalized_stats, interrupted)` when streaming finishes.
- Put common stats in `ChatResult.stats`: `prompt_tokens`, `completion_tokens`,
  `completion_duration_ns`, and `finish_reason`.
- Translate provider/network errors to `RuntimeError`; do not expose raw credentials in
  messages or logs.

The chat call is synchronous from the adapter's point of view and runs in NEXUS's request
worker. A streaming adapter should call `on_piece` as text arrives and close its response
when it stops or fails.

## Registration steps

1. Add `providers/<name>.py` and implement the contract above. Import `ChatResult` from
   `providers`.
2. If the provider needs ordinary settings, add safe defaults to `agent.DEFAULT_CONFIG`
   and register their names in the provider's `settings=(...)` tuple. The UI uses that
   tuple to show only the selected provider's settings.
3. Import the adapter at the bottom of `providers/__init__.py`, then call
   `register_provider("stable-id", "Display name", Adapter, settings=(...))`.
4. Update setup documentation and add adapter-specific tests using a fake HTTP/API
   response. Never require live credentials in tests.

The UI's provider and model controls are populated from the registry automatically. Keep
API keys out of `DEFAULT_CONFIG` and workspace config: read them from an environment
variable or Windows Credential Manager. Do not save secrets in chat history.

Claude and Codex support still needs a concrete integration choice (for example, a direct
service API or an installed local CLI). This adapter seam prepares the app for that work;
it does not bundle either integration or its authentication flow.

## Ollama and live learning

Ollama is the default generative responder. It uses the local streaming API at
http://127.0.0.1:11434 by default and lists models already installed with Ollama. Install a model
locally before selecting it in NEXUS.

The optional live-learning setting is off by default. When enabled, completed Ask turns
without project context and approved code edits that pass checks are stored in local app data and
applied to the from-scratch checkpoint by a background worker. File attachments and project-context
Ask turns are skipped. One low-rate optimizer update is made per exchange with short answer
sequences and curated replay examples. The examples are also included in a batch-training run. This is a small experimental model and may learn errors from Ollama; the setting is easy
to disable and should stay off while the computer is hot.
