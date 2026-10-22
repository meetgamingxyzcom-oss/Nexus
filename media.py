"""Local media generation helpers for NEXUS.

Image inference is delegated to an already-installed local ComfyUI instance.
NEXUS never downloads models or sends prompts to a hosted media API.
"""

from __future__ import annotations

import json
import pathlib
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse


class MediaError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class MediaService:
    """Small local-only ComfyUI client with background image jobs."""

    def __init__(self, output_dir: pathlib.Path, base_url: str = "http://127.0.0.1:8188",
                 activity_lock: Optional[object] = None):
        self.output_dir = pathlib.Path(output_dir)
        self.base_url = self._validate_base_url(base_url)
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._lock = threading.RLock()
        self._active = False
        self._activity_lock = activity_lock
        self._jobs: Dict[str, dict] = {}

    @staticmethod
    def _validate_base_url(value: str) -> str:
        parsed = urlparse(str(value or ""))
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or
                parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("NEXUS_COMFYUI_URL must point to a local ComfyUI HTTP server")
        return f"http://{parsed.netloc}{parsed.path.rstrip('/')}"

    def _request(self, path: str, *, data: Optional[bytes] = None, timeout: float = 2.5,
                 limit: int = 2_000_000) -> bytes:
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with self._opener.open(req, timeout=timeout) as response:
                result = response.read(limit + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(4000).decode("utf-8", errors="replace")
            raise MediaError(f"Local image engine returned HTTP {exc.code}: {detail[:500]}", 502)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise MediaError("Local ComfyUI is not available. Start it on this computer, then try again.", 503) from exc
        if len(result) > limit:
            raise MediaError("Local image engine returned a file that is too large.", 502)
        return result

    @staticmethod
    def _json(data: bytes) -> dict:
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise MediaError("Local image engine returned an invalid response.", 502) from exc
        if not isinstance(value, dict):
            raise MediaError("Local image engine returned an invalid response.", 502)
        return value

    def _models(self) -> list:
        data = self._json(self._request("/models/checkpoints", timeout=2.0, limit=500_000))
        return [str(item) for item in data if isinstance(item, str)] if isinstance(data, list) else []

    def status(self) -> dict:
        try:
            models = self._models()
            return {"connected": True, "ready": bool(models), "models": models,
                    "message": "Ready" if models else "ComfyUI is running, but no checkpoint is installed."}
        except MediaError as exc:
            return {"connected": False, "ready": False, "models": [], "message": exc.message}

    def start(self, prompt: str, width: int, height: int, checkpoint: str = "") -> dict:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise MediaError("Describe the image you want first.")
        if len(prompt) > 3000:
            raise MediaError("Image prompts are limited to 3,000 characters.")
        if (width, height) not in {(512, 512), (512, 768), (768, 512)}:
            raise MediaError("Choose a supported image shape.")
        models = self._models()
        if not models:
            raise MediaError("ComfyUI is running, but it has no image checkpoint. Add a checkpoint and try again.", 409)
        checkpoint = str(checkpoint or "").strip()
        if checkpoint and checkpoint not in models:
            raise MediaError("That image checkpoint is no longer available. Refresh the media studio and choose another.", 409)
        checkpoint = checkpoint or models[0]
        job_id = self._new_job("Waiting for the local image engine…")
        threading.Thread(target=self._run, args=(job_id, prompt, width, height, checkpoint), daemon=True).start()
        return {"id": job_id, "state": "queued"}

    def _new_job(self, message: str) -> str:
        with self._lock:
            if self._active:
                raise MediaError("Another media generation is already running. Wait for it to finish.", 409)
            if self._activity_lock is not None and not self._activity_lock.acquire(blocking=False):
                raise MediaError("Wait for the current NEXUS chat or code run to finish before generating media.", 409)
            self._active = True
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = {"id": job_id, "state": "queued", "message": message}
            for old_id in list(self._jobs)[:-20]:
                self._jobs.pop(old_id, None)
        return job_id

    def start_video(self, prompt: str, workflow: dict) -> dict:
        prompt = str(prompt or "").strip()
        if not prompt:
            raise MediaError("Describe the video you want first.")
        if len(prompt) > 3000:
            raise MediaError("Video prompts are limited to 3,000 characters.")
        if isinstance(workflow, dict) and isinstance(workflow.get("prompt"), dict):
            workflow = workflow["prompt"]
        if not isinstance(workflow, dict) or not 1 <= len(workflow) <= 2000:
            raise MediaError("Choose a ComfyUI workflow exported in API format (up to 2,000 nodes).")
        try:
            if len(json.dumps(workflow, ensure_ascii=False).encode("utf-8")) > 1_500_000:
                raise MediaError("Video workflow files are limited to 1.5 MB.")
        except (TypeError, ValueError) as exc:
            raise MediaError("That video workflow is not valid JSON.") from exc
        graph = json.loads(json.dumps(workflow))
        for node in graph.values():
            if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
                raise MediaError("That is not an API-format ComfyUI workflow. Export it using ComfyUI's API format.")
        if not self._apply_video_prompt(graph, prompt):
            raise MediaError("I couldn't find a positive prompt field in this workflow. Use an API workflow with a text-encode node.")
        self._models()  # Confirm the local service is reachable before accepting the job.
        job_id = self._new_job("Sending your workflow to the local video engine…")
        threading.Thread(target=self._run_video, args=(job_id, graph), daemon=True).start()
        return {"id": job_id, "state": "queued"}

    @staticmethod
    def _apply_video_prompt(graph: dict, prompt: str) -> bool:
        """Update positive text encoders connected to sampler positive inputs."""
        positive_roots = []
        for node in graph.values():
            inputs = node.get("inputs", {})
            kind = node.get("class_type", "").lower()
            if "sampl" not in kind and "guider" not in kind:
                continue
            for key in ("positive", "conditioning", "positive_cond"):
                link = inputs.get(key)
                if isinstance(link, list) and link and isinstance(link[0], str):
                    positive_roots.append(link[0])
        positive_nodes = set()
        pending = positive_roots[:]
        while pending and len(positive_nodes) < 2000:
            node_id = pending.pop()
            if node_id in positive_nodes or node_id not in graph:
                continue
            positive_nodes.add(node_id)
            for value in graph[node_id].get("inputs", {}).values():
                if isinstance(value, list) and value and isinstance(value[0], str):
                    pending.append(value[0])
        changed = False
        prompt_keys = {"text", "prompt", "positive", "prompt_text", "text_g", "text_l", "positive_prompt"}
        for node_id, node in graph.items():
            inputs = node.get("inputs", {})
            kind = node.get("class_type", "").lower()
            meta = node.get("_meta", {})
            label = str(meta.get("title", "") if isinstance(meta, dict) else "").lower()
            is_encoder = any(word in kind for word in ("textencode", "cliptext", "prompt"))
            if not is_encoder or (positive_nodes and node_id not in positive_nodes):
                continue
            if not positive_nodes and any(word in label for word in ("negative", "uncond")):
                continue
            for key, value in list(inputs.items()):
                if key.lower() in prompt_keys and isinstance(value, str):
                    inputs[key] = prompt
                    changed = True
        return changed

    def get_job(self, job_id: str) -> dict:
        if not isinstance(job_id, str) or len(job_id) != 32 or any(ch not in "0123456789abcdef" for ch in job_id):
            raise MediaError("Image job not found.", 404)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise MediaError("Image job not found.", 404)
            return dict(job)

    def file(self, file_id: str) -> Tuple[pathlib.Path, str, str]:
        if not isinstance(file_id, str) or len(file_id) != 32 or any(ch not in "0123456789abcdef" for ch in file_id):
            raise MediaError("Generated media not found.", 404)
        extensions = {".png": "image/png", ".mp4": "video/mp4", ".mkv": "video/x-matroska",
                      ".webm": "video/webm", ".gif": "image/gif"}
        path = next((self.output_dir / (file_id + ext) for ext in extensions
                     if (self.output_dir / (file_id + ext)).is_file()), None)
        if path is None:
            raise MediaError("Generated media not found.", 404)
        try:
            with path.open("rb") as source:
                signature = source.read(16)
        except OSError as exc:
            raise MediaError("Generated media not found.", 404) from exc
        valid = {
            ".png": signature.startswith(b"\x89PNG\r\n\x1a\n"),
            ".mp4": b"ftyp" in signature[:16],
            ".mkv": signature.startswith(b"\x1aE\xdf\xa3"),
            ".webm": signature.startswith(b"\x1aE\xdf\xa3"),
            ".gif": signature.startswith((b"GIF87a", b"GIF89a")),
        }
        if not valid.get(path.suffix, False):
            raise MediaError("Generated media file is incomplete or invalid.", 500)
        return path, extensions[path.suffix], path.suffix

    def _set_job(self, job_id: str, **updates) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(updates)

    def _workflow(self, prompt: str, width: int, height: int, checkpoint: str, seed: int) -> dict:
        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, blurry, distorted", "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "5": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 20, "cfg": 7.0,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "NEXUS", "images": ["6", 0]}},
        }

    def _run(self, job_id: str, prompt: str, width: int, height: int, checkpoint: str) -> None:
        workflow = self._workflow(prompt, width, height, checkpoint, secrets.randbelow(2**53))
        self._run_workflow(job_id, workflow, "image")

    def _run_video(self, job_id: str, workflow: dict) -> None:
        self._run_workflow(job_id, workflow, "video")

    def _run_workflow(self, job_id: str, workflow: dict, media_kind: str) -> None:
        try:
            self._set_job(job_id, state="running", message="Generating on this computer…")
            client_id = uuid.uuid4().hex
            request = {"prompt": workflow, "client_id": client_id}
            response = self._json(self._request("/prompt", data=json.dumps(request).encode("utf-8"),
                                                timeout=5.0, limit=500_000))
            prompt_id = str(response.get("prompt_id", ""))
            if not prompt_id:
                details = response.get("node_errors")
                label = "video" if media_kind == "video" else "image"
                raise MediaError(f"ComfyUI rejected the {label} workflow." + (f" {details}" if details else ""), 502)
            deadline = time.monotonic() + 600
            history = None
            while time.monotonic() < deadline:
                time.sleep(1.0)
                data = self._json(self._request("/history/" + urllib.parse.quote(prompt_id, safe=""),
                                                timeout=3.0, limit=2_000_000))
                history = data.get(prompt_id)
                if isinstance(history, dict) and history.get("outputs"):
                    break
                if isinstance(history, dict):
                    messages = history.get("status", {}).get("messages", [])
                    if any(isinstance(item, list) and item and item[0] == "execution_error" for item in messages):
                        label = "video" if media_kind == "video" else "image"
                        raise MediaError(f"ComfyUI could not generate this {label}. Check its console for model errors.", 502)
                self._set_job(job_id, message="Generating on this computer… this can take a few minutes.")
            if not isinstance(history, dict) or not isinstance(history.get("outputs"), dict):
                label = "video" if media_kind == "video" else "image"
                raise MediaError(f"{label.capitalize()} generation timed out. Check the local ComfyUI window.", 504)
            asset = None
            allowed_video = {".mp4", ".mkv", ".webm", ".gif"}
            for output in history["outputs"].values():
                if not isinstance(output, dict):
                    continue
                keys = ("videos", "gifs", "files", "images") if media_kind == "video" else ("images",)
                for key in keys:
                    items = output.get(key, [])
                    if isinstance(items, dict):
                        items = [items]
                    for item in items if isinstance(items, list) else []:
                        if not isinstance(item, dict) or not item.get("filename"):
                            continue
                        ext = pathlib.Path(str(item["filename"])).suffix.lower()
                        if media_kind == "video" and ext in allowed_video:
                            asset = (item, ext, "video")
                            break
                        if media_kind == "image" and ext == ".png":
                            asset = (item, ext, "image")
                            break
                    if asset:
                        break
                if asset:
                    break
            if not asset:
                label = "video" if media_kind == "video" else "image"
                raise MediaError(f"ComfyUI finished without returning a supported {label} file.", 502)
            asset_info, extension, output_kind = asset
            params = urllib.parse.urlencode({"filename": asset_info.get("filename", ""),
                                             "subfolder": asset_info.get("subfolder", ""),
                                             "type": asset_info.get("type", "output")})
            self.output_dir.mkdir(parents=True, exist_ok=True)
            file_id = uuid.uuid4().hex
            self._download_asset(params, file_id, extension)
            label = "Video" if output_kind == "video" else "Image"
            self._set_job(job_id, state="done", message=f"{label} ready.", file_id=file_id,
                          media_type=output_kind, extension=extension)
        except MediaError as exc:
            self._set_job(job_id, state="error", message=exc.message)
        except Exception as exc:  # Keep one failed media run from wedging the service.
            label = "video" if media_kind == "video" else "image"
            self._set_job(job_id, state="error", message=f"{label.capitalize()} generation failed: {type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._active = False
            if self._activity_lock is not None:
                self._activity_lock.release()

    def _download_asset(self, params: str, file_id: str, extension: str) -> None:
        if extension not in {".png", ".mp4", ".mkv", ".webm", ".gif"}:
            raise MediaError("ComfyUI returned an unsupported media format.", 502)
        path = self.output_dir / (file_id + extension)
        req = urllib.request.Request(self.base_url + "/view?" + params)
        try:
            with self._opener.open(req, timeout=30.0) as response, path.open("wb") as destination:
                size = 0
                signature = b""
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 150_000_000:
                        raise MediaError("Generated video is larger than the 150 MB limit.", 413)
                    if len(signature) < 16:
                        signature += chunk[:16 - len(signature)]
                    destination.write(chunk)
            if size == 0:
                raise MediaError("ComfyUI returned an empty media file.", 502)
            valid = {
                ".png": signature.startswith(b"\x89PNG\r\n\x1a\n"),
                ".mp4": b"ftyp" in signature[:16],
                ".mkv": signature.startswith(b"\x1aE\xdf\xa3"),
                ".webm": signature.startswith(b"\x1aE\xdf\xa3"),
                ".gif": signature.startswith((b"GIF87a", b"GIF89a")),
            }
            if not valid.get(extension, False):
                raise MediaError("ComfyUI returned an invalid media file.", 502)
        except MediaError:
            path.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            path.unlink(missing_ok=True)
            raise MediaError("Could not download the generated file from local ComfyUI.", 502) from exc
