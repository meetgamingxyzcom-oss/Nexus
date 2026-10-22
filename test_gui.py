"""
Offline tests for the web UI server (gui.py). A fake model replaces a live model and
a real HTTP server runs on a free local port.

    python -m unittest discover -s tests -v
"""
import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import gui  # noqa: E402


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def ensure_model(self):
        pass

    def list_models(self):
        return ["fake:latest"]

    def chat_stream(self, messages, num_ctx, num_predict, on_piece):
        text = self.replies.pop(0)
        for i in range(0, len(text), 7):
            on_piece(text[i:i + 7])
        return agent.ChatResult(text, {"prompt_eval_count": 50, "eval_count": 20, "eval_duration": 10**9})


class Api:
    def __init__(self, port, token):
        self.base, self.token = f"http://127.0.0.1:{port}", token

    def call(self, path, body=None, headers=None, raw=False):
        h = {"X-Nexus-Token": self.token}
        h.update(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=h)
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = r.read().decode()
            return payload if raw else json.loads(payload)

    def events(self, job):
        out = []
        req = urllib.request.Request(f"{self.base}/api/events?job={job}&from=0&t={self.token}")
        with urllib.request.urlopen(req, timeout=15) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: "):
                    out.append(json.loads(line[6:]))
        return out


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())
        (self.root / "hello.py").write_text('print("Hello, world!")\n', encoding="utf-8")
        self.fake = FakeClient([])
        self.app = gui.App("tok123", lambda root, cfg: gui.GuiEngine(root, cfg, client=self.fake))
        self.server = gui.serve(self.app, "127.0.0.1", 18765)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.api = Api(self.app.port, "tok123")
        self.api.call("/api/workspace", {"path": str(self.root)})

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def wait_idle(self):
        for _ in range(100):
            if not self.app.run_lock.locked():
                return
            time.sleep(0.05)

    def test_security(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(f"{self.api.base}/api/state", timeout=5)
        self.assertEqual(ctx.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.api.call("/api/pin", {"path": "hello.py", "pinned": True}, {"Origin": "http://evil.example"})
        self.assertEqual(ctx.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.api.call("/api/file?path=../secret.txt")
        self.assertEqual(ctx.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.api.call("/api/state", headers={"Host": "evil.example"})
        self.assertEqual(ctx.exception.code, 403)

    def test_index_gets_token(self):
        page = self.api.call("/", raw=True)
        self.assertIn('content="tok123"', page)
        self.assertNotIn("__TOKEN__", page)

    def test_state_files_and_pin(self):
        st = self.api.call("/api/state")
        self.assertEqual(st["workspace"], str(self.root.resolve()))
        files = self.api.call("/api/files")
        self.assertEqual([f["rel"] for f in files["files"]], ["hello.py"])
        self.assertEqual(self.api.call("/api/pin", {"path": "hello.py", "pinned": True})["pinned"], ["hello.py"])
        self.assertIn("Hello", self.api.call("/api/file?path=hello.py")["content"])

    def test_ask_streams_tokens(self):
        self.fake.replies = ["It prints a greeting."]
        job = self.api.call("/api/chat", {"prompt": "What does hello.py do?", "mode": "auto"})["job"]
        events = self.api.events(job)
        kinds = [e["type"] for e in events]
        self.assertIn("context", kinds)
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "It prints a greeting.")
        self.assertEqual(kinds[-1], "done")

    def test_edit_review_apply_undo(self):
        self.fake.replies = [
            "SUMMARY: added greet\n=== EDIT: hello.py ===\n<<<<<<< SEARCH\nprint(\"Hello, world!\")\n=======\n"
            "def greet(n):\n    return n\n\n\nprint(greet(\"x\"))\n>>>>>>> REPLACE\n=== END ===\n"]
        job = self.api.call("/api/chat", {"prompt": "Add a greet function to hello.py", "mode": "edit"})["job"]
        result = {}
        thread = threading.Thread(target=lambda: result.update(ev=self.api.events(job)))
        thread.start()
        proposal = None
        for _ in range(100):
            for ev in self.app.jobs[job].events:
                if ev["type"] == "proposal":
                    proposal = ev
            if proposal:
                break
            time.sleep(0.05)
        self.assertIsNotNone(proposal)
        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), 'print("Hello, world!")\n')
        with self.assertRaises(urllib.error.HTTPError):  # stale proposal id is refused
            self.api.call("/api/decision", {"job": job, "proposal": "nope", "approve": ["hello.py"]})
        self.api.call("/api/decision", {"job": job, "proposal": proposal["id"], "approve": ["hello.py"]})
        thread.join(15)
        kinds = [e["type"] for e in result["ev"]]
        self.assertIn("applied", kinds)
        self.assertIn("checks", kinds)
        self.assertIn("def greet", (self.root / "hello.py").read_text(encoding="utf-8"))
        self.wait_idle()
        hist = self.api.call("/api/history")["items"]
        self.assertEqual(len(hist), 1)
        self.assertTrue(self.api.call("/api/undo", {})["ok"])
        self.assertEqual((self.root / "hello.py").read_text(encoding="utf-8"), 'print("Hello, world!")\n')

    def test_reject_writes_nothing(self):
        self.fake.replies = ["SUMMARY: x\n=== FILE: new.py ===\nx = 1\n=== END ===\n"]
        job = self.api.call("/api/chat", {"prompt": "create new.py", "mode": "edit"})["job"]
        t = threading.Thread(target=lambda: self.api.events(job))
        t.start()
        for _ in range(100):
            props = [e for e in self.app.jobs[job].events if e["type"] == "proposal"]
            if props:
                break
            time.sleep(0.05)
        self.api.call("/api/decision", {"job": job, "proposal": props[0]["id"], "approve": []})
        t.join(15)
        self.assertFalse((self.root / "new.py").exists())

    def test_busy_returns_409(self):
        self.app.run_lock.acquire()
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.api.call("/api/chat", {"prompt": "hi", "mode": "ask"})
            self.assertEqual(ctx.exception.code, 409)
        finally:
            self.app.run_lock.release()

    def test_settings(self):
        r = self.api.call("/api/settings", {"key": "num_ctx", "value": "16384"})
        self.assertEqual(r["config"]["num_ctx"], 16384)
        with self.assertRaises(urllib.error.HTTPError):
            self.api.call("/api/settings", {"key": "bogus", "value": 1})


if __name__ == "__main__":
    unittest.main()
