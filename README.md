# NEXUS Code 2.0: C launcher + Python engine with provider adapters

A local-first project workspace for Windows. The C program (`nexus.exe`) is the terminal
interface. The Python engine (`agent.py`) contains the workspace, build/run and safety
features. Its built-in NEXUS Brain Lite is a small local question matcher with curated
references for coding, graphics, mathematics, physics, chemistry, behaviorism, English, and
Hindi. It has no generative model or pretrained weights: it returns saved explanations rather
than inventing new ones. Saved examples teach the matcher, not a language model. A bounded math
engine remains available in Brain Lite. **Ollama (local)** is the default generative responder and
streams from an Ollama server on this computer. The separate **CodeAI Personal Model** is a small
character-level model trained from random weights; its limits are described below.

## What changed from version 1

| Area | Version 1 | Version 2 |
|---|---|---|
| Context | Sent up to 30 whole files, often overwhelming a small model | Project text is ranked and bounded; Brain Lite can return a relevant excerpt but does not reason over the whole project |
| Edit format | Whole files escaped inside JSON | Brain Lite does not propose edits; the workspace editor supports reviewed manual changes |
| Safety | Wrote immediately | Manual and model-proposed changes use path checks, review, backups, and undo |
| Undo | Manual copy from backup folder | `/undo` and `/history`, one change set per request, warns if you edited the file afterwards |
| Feedback | Waited silently up to 300 s | Brain Lite references return immediately |
| Mistakes | Reported syntax errors, nothing more | Syntax checks and test output are available |
| Memory | None | Chats are saved per workspace |
| Launcher | Used `system()` with the current folder, a hard-coded `python`, ASCII-only input | `CreateProcess` with correct quoting, finds `agent.py` beside the exe, finds Python itself, UTF-8 input, colors, multi-line input, `/ws`, `/cls` |
| Config | Edit the source | `nexus.json`, per-workspace `.nexus/config.json`, `/set`, `/model` |

## Requirements

- Windows 10/11
- Python 3.9 or newer
- NumPy is required to train or run the personal generative model (`python -m pip install -r model/requirements.txt`)
- GCC (MSYS2 MinGW-w64 UCRT64 recommended) to build `nexus.exe`
- Ollama is optional to install; the default generative provider expects Ollama and a locally downloaded model. NEXUS Brain Lite still works without a model server, model download, GPU, or external AI account.

## Setup

1. Put all files in one folder, e.g. `C:\Users\YOU\Documents\NEXUS_Code`
   (`nexus.exe`, `agent.py`, `main.c`, `build.bat`, `run.bat`, `nexus.json`, and for the web UI `gui.py`, `runner.py`, `gui.bat`, the `providers` folder and the `ui` folder; `runner.py` is also needed by `agent.py` for `/run`).
2. Rename `_gitignore` to `.gitignore` if you use git (optional).
3. Start `gui.bat` to open the website. Ollama is the default responder; NEXUS Brain Lite remains available offline. Install Ollama and pull the default model llama3.2 before starting chat.
4. Make a **test** workspace:

```powershell
mkdir "$HOME\Desktop\NEXUS_TestProject"
Set-Content "$HOME\Desktop\NEXUS_TestProject\hello.py" 'print("Hello, world!")'
```

5. Brain Lite includes a built-in library of coding, graphics, science, math, writing, and Hindi
   references. Sign in as the owner, choose **NEXUS Brain** in the Agent provider menu, then open
   Settings → Brain training to browse those topics or add project-specific examples. Similar
   questions return the saved answer.

## Train the personal generative model from scratch

CodeAI includes a small character-level recurrent neural language model with random initialization. It uses no
pretrained weights or model server. Training creates its weights in
`model/checkpoints/personal_char_lstm.npz` inside the project folder. Because this project is in
OneDrive, your OneDrive sync settings may also sync that checkpoint. The starter
corpus is built from Brain Lite's curated examples and selected CodeAI documentation and
source files. That is a small seed corpus, not training on every language or every subject.

Install NumPy into the same Python environment used to start CodeAI, then train:

```powershell
python -m pip install -r model/requirements.txt
python model/train.py --steps 20000
```

To add more training text, put openly licensed or personally owned `.txt`, `.md`, or source files
under `model/training_data/` before training. Do not put secrets or private data there. Continue
training from the saved checkpoint with `python model/train.py --resume --steps 20000`; each run
keeps intermediate checkpoints.

The starter model has about 136,000 parameters, a 192-byte prompt context, and a 1,536-byte answer
training limit (about 384 ordinary text tokens). Its trainer learns answer bytes conditioned on
their question and weights curated Q&A examples ten times more than raw source excerpts.
High-confidence matches use saved local reference answers; unreadable or incomplete free-form
drafts are suppressed, while other generated replies can still be incorrect. It is not an all-language
expert. Expanding its knowledge needs much larger, legally usable corpora and substantially more
compute. In the Agent panel, choose
**CodeAI Personal Model** to use its trained checkpoint; Brain Lite remains a separate reference
matcher.

`agent.py` must stay beside `nexus.exe`. You can start the launcher from any folder.

## NEXUS website and workspace

NEXUS opens as a professional website with About, Workspace, How it works, Privacy, FAQ, and Contact sections. Sign in to open the project workspace at `/workspace`. The website and workspace use the same engine, so the existing safety features still apply (path checks, review before writing, backups, undo, and syntax checks).

```powershell
gui.bat                                  # open the NEXUS website in your browser
gui.bat "%USERPROFILE%\Desktop\NEXUS_TestProject"
```

Or run `python gui.py --workspace "C:\path\to\project"` (`--port 9000`, `--no-browser` also work).
`gui.bat` needs no build step; `nexus.exe` is only for the terminal version.

What you get:

- **Local accounts**: the first account is an admin. An existing single-account installation is migrated to a standard account and receives a separate `nexus-admin` login; admins can create standard accounts from Settings. Standard accounts can use the workspace, while provider, model, build, and workspace-wide advanced controls are reserved for admins. Passwords are stored only as salted PBKDF2 hashes in your Windows local app data folder. If you cannot sign in, use **Can't sign in? Reset local accounts** on the sign-in screen; NEXUS keeps a backup of the old account file and lets you create a new administrator account.
- **Browser website**: open the NEXUS address on Windows, macOS, phones, or tablets and sign in to the same server account. Shared chats and projects remain on that server. For a public HTTPS website, follow [`CLOUD_SETUP.md`](CLOUD_SETUP.md).
- **Contact section**: self-hosted deployments can set the optional `NEXUS_CONTACT_EMAIL` in `.env` to show a support email on the website. If it is empty, the website directs visitors to the person who operates that NEXUS server.
- **Chat timeline** with streaming answers, safe markdown rendering and copyable code blocks.
- **File attachments**: choose files with **Attach** or drag text/code files into the chat. Up to 10 files can be added (80 KB each, 300 KB total); supported text is included with the message and is not saved as a separate upload. Binary images are handled separately by Media Studio.
- **Local AI providers**: Ollama streams generative replies from a model installed on your computer. NEXUS Brain Lite and the from-scratch CodeAI Personal Model remain available.
- **Background learning**: an admin can enable *Train NEXUS from Ollama chats* in Settings. Completed Ask turns without attachments or project context, and approved code edits that pass checks, are saved in local app data and receive one small background update. It is off by default to reduce heat. Ollama can make mistakes, which NEXUS may learn.
- **Graphics and media**: projects can include text-based SVG artwork and graphics code using HTML,
  CSS, canvas, and shaders. Media Studio can generate PNG pictures with a local ComfyUI image
  checkpoint, run a ComfyUI API-format video workflow, or animate a still image into a short browser
  WebM clip. The still-image animation pans and zooms; it does not synthesize new video frames.
- **A place to write**: click any file and the right-hand pane is a real editor. Type directly, press Ctrl+S to save
  (Tab indents, Enter keeps the indentation). **New** in the sidebar creates a file. Saves go through the same safety net
  as the model: path and file-type checks, a backup you can undo in the Changes tab, and a syntax check shown under the editor.
  If the file changed on disk while you were typing (for example NEXUS edited it), you are asked before anything is overwritten.
- **Code notepad**: a private, account-specific scratch space that autosaves. With a workspace open, choose **Run in terminal**
  to run the current draft in the built-in terminal without saving it as a project file. **Save a copy to project** is still available.
- **Calm look**: warm paper or soft charcoal (follows your system, switch with the half-moon button), one muted accent,
  serif reading text, no glow or flashing.
- **File explorer** with search, and a pin to always include a file in the context.
- **`@` mentions**: type `@` in the box to autocomplete a file name.
- **Changes tab** listing every applied change set with a one-click **Undo** (warns if you edited the file since).
- **Live status**: context usage meter, warnings when a reply is cut off, and a collapsible raw provider reply for debugging.
- **Agent panel** under the file search: choose Brain Lite or CodeAI Personal Model. A Tests button
  (runs your `test_command` and shows its output), settings, light/dark theme and Quit.
- **Shortcuts**: Enter send, Shift+Enter new line, Esc stop, Ctrl+B sidebar, Ctrl+K focus the box.
  Typing `/undo`, `/clear`, `/test`, `/ask ...`, `/edit ...` or `/settings` also works.

Security of the local interface: it listens on `127.0.0.1` only, every request needs a random per-run token
and a signed local-account session, the Host and Origin headers are checked (which blocks other websites and
DNS-rebinding), the page has a strict Content-Security-Policy, and model output is never inserted as HTML. The cloud setup keeps NEXUS behind Caddy, requires an exact HTTPS origin and a private first-account setup code, limits the workspace root, and sets secure cookies.
The account file lives at `%LOCALAPPDATA%\Nexus\account.json`; the sign-in recovery action moves it to a timestamped backup before resetting local accounts. Local malware
on your PC is still out of scope.
Only one request runs at a time; Stop (or Esc) cancels the active response.

The training/inference code is in `model/train.py` and `providers/personal.py`. See
[`providers/README.md`](providers/README.md) for the provider boundary. Brain Lite's saved examples
remain separate from model weights; the trainer reads the built-in examples as seed training text.

## Compile and run your code (new)

Press **Build & Run** (header, editor, or **F5**) and NEXUS identifies the source language from its extension, then builds or runs it with a matching tool already on your PATH. Plain-text source can be detected by its contents; if the language is unclear, NEXUS asks you to choose one. It runs the file in an
**inbuilt terminal** at the bottom of the web interface. Output streams live, you can type input to the
program in the box under the output, and **Stop** ends it. **Open in my terminal** instead opens a real Windows
console window that compiles and runs the file and stays open when it finishes.

| Language or file | What Build & Run does |
|---|---|
| Python | Current Python interpreter |
| C / C++ | GCC or Clang; helper source files are linked automatically |
| JavaScript / JSX / TypeScript | Node.js, or `tsx` / `ts-node` for JSX and TypeScript |
| Java / Kotlin | `javac` + `java`, or `kotlinc` + `java` |
| Go / Rust / Swift | `go`, `rustc`, or `swiftc` |
| C# | `csc` or `mcs` |
| Ruby / PHP / PowerShell / Bash / Lua / Perl | Matching runtime from PATH |
| `.bat` `.cmd` | Windows command processor (read scripts first) |
| `.html` `.svg` | opens in your web browser |

NEXUS does not download or install toolchains. If the matching compiler/runtime is missing,
the message explains which tool is needed. **Preferred code language** in the composer controls
the language NEXUS creates or rewrites; Auto follows existing project files and asks when unsure.

## Create pictures and videos

Choose **Media** beside the chat attachment button. Picture generation uses ComfyUI already running
on this computer and a local checkpoint; the built-in workflow uses a 512-pixel image and 20 steps.
NEXUS does not download image or video weights. Generated files are saved under
`generated_media` in the NEXUS data directory.

For AI video generation, choose a ComfyUI workflow exported in **API format**. The workflow must
include a positive text-encode node connected to a sampler, plus its own compatible video model and
any required custom nodes. NEXUS inserts your prompt into the connected positive text encoder and
sends the workflow only to the local ComfyUI server. This keeps model-specific video pipelines
flexible without assuming one video model or downloading large files. A ComfyUI video workflow can
use its Save Video output to return MP4, WebM, or MKV. If you do not have a video model/workflow,
**Animate still** makes a four-second pan-and-zoom WebM in the browser from a generated or selected
picture; it does not create new AI-generated frames. ComfyUI custom nodes can contact their own
services, so check the workflow and its nodes if local-only processing matters.

ComfyUI's default local address is `http://127.0.0.1:8188`. If it uses another local port, set
`NEXUS_COMFYUI_URL` to that localhost address before starting NEXUS. NEXUS accepts only loopback
addresses for this integration and makes no hosted image-service calls.

Details that matter:

- **Prompts show up before you type.** Normally a C program writing to a pipe holds its output back, so
  `printf("Name: ")` would not appear until the program ended. For the inbuilt terminal NEXUS compiles with a tiny generated
  header (`.nexus\build\nexus_unbuf.h`) that turns buffering off. Your source files are never touched.
- **Build errors stay visible.** If the build fails or the program exits with an error, NEXUS shows the terminal output. Brain Lite cannot diagnose it.
- **Run after changes** (checkbox in the terminal bar, or `auto_run` in Settings) starts the program by itself after AI edits are applied or you
  press Save, provided the syntax checks passed. It is **off by default** and asks you to confirm when you switch it on,
  because it will run code the model wrote. Only enable it for projects you trust.
- **Safety:** a program only starts when you press Build & Run, use F5, or have switched on auto-run yourself. The model has no way to start one.
  Commands are built from fixed templates (or your own `run_command`) as argument lists, never through a shell, so a file name
  cannot inject a command. The first Run in each browser session asks for confirmation. Programs run with your permissions,
  from the workspace folder, with a time limit (`run_timeout`, default 300 s) and a 2 MB output cap.
- **Terminal version:** `/run main.c` (or just `/run` for the file you just edited) detects the file type and compiles or runs it in the same console, so the
  program can read your keyboard directly. With `auto_run` on it also runs after applied edits.
- **Settings:** grouped by AI, Build & Run, Editing and provider. Admins can tune provider/model, custom build commands, arguments, flags, automatic actions and workspace limits; standard accounts see only everyday preferences. `run_command` is an advanced override (with `{file} {rel} {dir} {name} {root}`); normally language detection selects the toolchain. Also available: `run_args`, `build_flags` (for example `-lm -O2`),
  `run_timeout`, `auto_run`.
- Needs `gcc` on PATH for C and C++ (the same MSYS2 install used to build `nexus.exe`). If it is missing you get a message saying so.

## Using it (terminal version)

```text
nexus> Explain the files in this workspace. Do not edit anything.
nexus> Add a function greet(name) to hello.py that returns a greeting, and call it
nexus> In @main.c, make the temp file name include the process id
nexus> ? why does run_agent need CreateProcess
nexus> Reply with only the word OK
```

- **Mode detection**: questions and "do not edit" requests are answered read-only.
  Everything else is an edit. Force a mode with `/ask ...`, `/edit ...` or a leading `?`.
- **Edits** appear as a colored diff. Answer `y` (all), `n` (none) or `e` (decide per file).
- **Multi-line requests**: end a line with `\` to continue it.
- **Cancel**: Ctrl+C stops generation; nothing is written from a cancelled reply.
- **Quick answers**: requests starting with "Reply with only ..." skip project context and
  return almost instantly.

### Commands

| Command | Purpose |
|---|---|
| `/help` | List commands |
| `/files` | Show which files are eligible and which are pinned |
| `/add <path>` / `/drop <path or all>` | Pin files so they are always in the context |
| `/undo` | Revert the most recent change set (also removes files NEXUS created) |
| `/history` | List change sets |
| `/test` | Run your configured `test_command` |
| `/fix` | Run the tests, then ask the model to fix failures |
| `/model [name]`, `/models` | Show or switch the model, list installed models |
| `/set [key value]` | Show or change settings (saved per workspace) |
| `/status` | Check the selected provider, model, memory, gcc |
| `/clear` | Forget conversation memory |
| `/ws <folder>`, `/cls`, `/exit` | Handled by the launcher |

### Settings (`nexus.json` or `/set`)

| Key | Default | Notes |
|---|---|---|
| `model` | `nexus-brain-lite` | The built-in question-matching model |
| `code_language` | `auto` | Preferred language for code NEXUS creates or rewrites; Auto follows the project and asks when unclear |
| `num_ctx` | 8192 | Maximum context budget used when project text is displayed |
| `num_predict` | 2048 | Max reply length. Raise to 4096 for bigger edits |
| `temperature` | 0.1 | Keep low for code |
| `history_turns` | 4 | Previous turns replayed to the model |
| `max_repair_attempts` | 2 | Automatic fix-ups after a failed check |
| `auto_apply` | false | Skip the confirmation prompt (script files still ask) |
| `test_command` | empty | Your own command, for example `python -m unittest` |

Settings load in this order: built-in defaults, `nexus.json` beside `agent.py`, and
`<workspace>\.nexus\config.json` (written by `/set` and `/model`). The selected provider and model
are preserved per workspace.

## Teaching NEXUS Brain

- The built-in library covers selected programming languages, graphics, math, physics, chemistry, behaviorism, English, and Hindi topics. It is a small reference collection, not a complete textbook or all-language model.
- The math engine supports arithmetic operators, percentages, powers, common functions (`sqrt`, `gcd`, `comb`, trigonometry, mean/median and others), and linear or quadratic equations in `x`.
- Add concise, correct question-and-answer pairs in **Settings → Brain training**.
- Write examples the way you expect to be asked. Similar wording matches more reliably.
- To correct a saved answer, train the same question again with the preferred answer.
- The Brain does not learn from normal conversations, and it will say when no example is close enough.
- Its answers come from built-in references and examples you add; it is not a generative model and cannot solve arbitrary coding tasks.

## What the agent can and cannot do

Can: return built-in reference answers, evaluate supported arithmetic, read supported source files
(up to 400 scanned, 80 KB each, honoring `.gitignore` and skipping `.git`, `node_modules`, build
folders and secret-looking files), show relevant text from a project, back up and undo file
changes, run built-in syntax checks, and run **your** test command.

Cannot: guarantee complete knowledge of every subject or language, inspect binary images visually,
edit files outside the workspace, edit `.git` or `.nexus`, or browse the web. Brain Lite cannot
invent custom code or make code edits.

Built-in checks: Python (compile), JSON (parse), C (`gcc`/`clang -fsyntax-only`),
C++ (`g++`/`clang++`), JavaScript (`node --check`). Missing tools are reported as SKIP.

Where state lives locally: `<workspace>\.nexus\` holds `backups\`, `session.json` and `config.json`. In the GUI, each conversation is saved separately as `.nexus/chats/<id>.json`; existing session history is carried into the first chat when you open a workspace. In the private cloud deployment, those workspace files and chat histories live in persistent server volumes; see [`CLOUD_SETUP.md`](CLOUD_SETUP.md) for setup and backups.
Add `.nexus/` to your project's `.gitignore`.

## Safety notes

- Work on a copy or a git repository. Backups help but are not version control.
- Obvious secrets (`api_key = ...`, `password: ...`) are redacted and files such as `.env`, `*.pem`,
  `id_rsa*` are never read, but do not keep credentials in the workspace anyway.
- Brain examples are stored locally in `%LOCALAPPDATA%\Nexus\brain` on Windows, or on the persistent
  server volume in a cloud deployment. Do not teach it passwords, API keys, or private information.
- This is still a starter tool, not a sandbox. Always read the diff.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `gcc was not found` | Install MSYS2, add `C:\msys64\ucrt64\bin` to PATH, open a new terminal |
| `Python 3 was not found` | Install Python from python.org with "Add to PATH", or `set NEXUS_PYTHON=C:\full\path\python.exe` |
| `I haven't learned an answer to that yet` | Add a matching question and preferred answer under Settings → Brain training |
| "The prompt filled the whole context window" | `/set num_ctx 16384`, or pin fewer files |
| "reply hit the token limit" | `/set num_predict 4096` or request a smaller change |
| SEARCH text not found | The model misquoted the file. NEXUS retries automatically; if it keeps failing, name the exact function or lines in your request |
| Garbled colors in an old console | `set NO_COLOR=1` before running |

## Developing

Offline tests (provider responses are faked):

```powershell
cd NEXUS_Code
python -m unittest discover -s tests -v
```

The suite has 48 tests. `tests/test_gui.py` starts the real web server with a fake model and exercises security checks, streaming, review/apply/undo, rejection and settings. Together they cover the edit parser, the matching strategies, path safety, mode detection, the context
budget, line-ending preservation, undo, and the self-repair loop.
