"""
Tests for runner.py: plans, compiling, running with captured output, typing input
into a running program, stopping, time limits, and the external-terminal script.
Needs gcc for the C tests (they are skipped if it is missing).

    python -m unittest discover -s tests -v
"""
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import runner  # noqa: E402

HAVE_GCC = bool(shutil.which("gcc") or shutil.which("clang"))


def cfg(**extra):
    c = dict(agent.DEFAULT_CONFIG)
    c.update(extra)
    return c


def run_collect(root, rel, config=None, control=None, on_event=None):
    config = config or cfg()
    plan = runner.plan_run(root, rel, config)
    events = []

    def emit(kind, **data):
        events.append({"type": kind, **data})
        if on_event:
            on_event(events[-1])

    runner.run_plan(plan, root, config, emit, control or runner.ProcessControl())
    return plan, events


def output_of(events):
    return "".join(e.get("text", "") for e in events if e["type"] == "out")


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())

    def test_python_plan(self):
        (self.root / "a.py").write_text("print(1)\n", encoding="utf-8")
        plan = runner.plan_run(self.root, "a.py", cfg())
        self.assertEqual(plan.kind, "python")
        self.assertEqual([s.name for s in plan.steps], ["run"])

    def test_unknown_and_unsafe(self):
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(runner.BuildError):
            runner.plan_run(self.root, "a.txt", cfg())
        for bad in ("../x.py", "missing.py", ".env", "C:/Windows/x.bat"):
            with self.assertRaises(runner.BuildError, msg=bad):
                runner.plan_run(self.root, bad, cfg())

    def test_custom_command_placeholders(self):
        (self.root / "t.txt").write_text("x", encoding="utf-8")
        plan = runner.plan_run(self.root, "t.txt", cfg(run_command="mytool {name} {rel}", run_args="--fast"))
        args = plan.steps[0].args
        self.assertEqual(args[:3], ["mytool", "t", "t.txt"])
        self.assertEqual(args[-1], "--fast")

    def test_html_opens_browser(self):
        (self.root / "p.html").write_text("<p>x</p>", encoding="utf-8")
        self.assertIsNotNone(runner.plan_run(self.root, "p.html", cfg()).browser)

    @unittest.skipUnless(HAVE_GCC, "gcc not installed")
    def test_c_needs_main_and_links_helpers(self):
        (self.root / "helper.c").write_text("int twice(int x){return x*2;}\n", encoding="utf-8")
        (self.root / "main.c").write_text('#include <stdio.h>\nint twice(int);\nint main(void){printf("%d\\n",twice(21));return 0;}\n', encoding="utf-8")
        with self.assertRaises(runner.BuildError):
            runner.plan_run(self.root, "helper.c", cfg())
        plan = runner.plan_run(self.root, "main.c", cfg())
        self.assertIn("helper.c", plan.note)
        self.assertEqual([s.name for s in plan.steps], ["compile", "run"])

    def test_pick_runnable_prefers_main(self):
        for name in ("zeta.py", "main.py", "alpha.py"):
            (self.root / name).write_text("print(1)\n", encoding="utf-8")
        self.assertEqual(runner.pick_runnable(self.root, ["zeta.py", "main.py", "alpha.py"], cfg()), "main.py")
        self.assertIsNone(runner.pick_runnable(self.root, ["nothing.xyz"], cfg()))


class RunTests(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())

    def test_python_success_and_failure(self):
        (self.root / "ok.py").write_text("print('hello from nexus')\n", encoding="utf-8")
        _, events = run_collect(self.root, "ok.py")
        self.assertIn("hello from nexus", output_of(events))
        self.assertTrue(events[-1]["ok"])
        (self.root / "bad.py").write_text("raise ValueError('boom')\n", encoding="utf-8")
        _, events = run_collect(self.root, "bad.py")
        self.assertIn("ValueError: boom", output_of(events))
        self.assertFalse(events[-1]["ok"])
        self.assertEqual(events[-1]["code"], 1)

    def test_run_args(self):
        (self.root / "a.py").write_text("import sys\nprint('ARGS', sys.argv[1:])\n", encoding="utf-8")
        _, events = run_collect(self.root, "a.py", cfg(run_args="one two"))
        self.assertIn("ARGS ['one', 'two']", output_of(events))

    def test_time_limit(self):
        (self.root / "slow.py").write_text("import time\nprint('start', flush=True)\ntime.sleep(30)\n", encoding="utf-8")
        started = time.time()
        _, events = run_collect(self.root, "slow.py", cfg(run_timeout=1))
        self.assertLess(time.time() - started, 15)
        self.assertTrue(events[-1]["stopped"])
        self.assertIn("limit", output_of(events))

    def test_stop_button(self):
        (self.root / "loop.py").write_text("import time\nprint('go', flush=True)\nwhile True: time.sleep(0.1)\n", encoding="utf-8")
        control = runner.ProcessControl()
        threading.Timer(1.0, control.kill).start()
        _, events = run_collect(self.root, "loop.py", control=control)
        self.assertTrue(events[-1]["stopped"])

    def test_typing_input_into_python(self):
        (self.root / "ask.py").write_text("name = input('Name: ')\nprint('Hi ' + name)\n", encoding="utf-8")
        control = runner.ProcessControl()
        seen = {"sent": False}

        def on_event(ev):
            if ev["type"] == "out" and "Name:" in ev["text"] and not seen["sent"]:
                seen["sent"] = True
                control.write("Ada\n")

        _, events = run_collect(self.root, "ask.py", control=control, on_event=on_event)
        self.assertIn("Hi Ada", output_of(events))

    @unittest.skipUnless(HAVE_GCC, "gcc not installed")
    def test_c_prompt_appears_before_input_and_reads_it(self):
        (self.root / "main.c").write_text(
            '#include <stdio.h>\nint main(void){char n[64];printf("Name: ");scanf("%63s",n);printf("Hello %s\\n",n);return 0;}\n',
            encoding="utf-8")
        control = runner.ProcessControl()
        seen = {"sent": False}

        def on_event(ev):  # if the prompt were buffered this would never fire and the test would hang
            if ev["type"] == "out" and "Name:" in ev["text"] and not seen["sent"]:
                seen["sent"] = True
                control.write("Bob\n")

        timer = threading.Timer(20, control.kill)
        timer.start()
        _, events = run_collect(self.root, "main.c", control=control, on_event=on_event)
        timer.cancel()
        self.assertTrue(seen["sent"], "prompt was buffered and never shown")
        self.assertIn("Hello Bob", output_of(events))
        phases = [e["name"] for e in events if e["type"] == "phase"]
        self.assertEqual(phases, ["compile", "run"])

    @unittest.skipUnless(HAVE_GCC, "gcc not installed")
    def test_c_compile_error_stops_before_run(self):
        (self.root / "main.c").write_text("int main(void){ return undefined_thing; }\n", encoding="utf-8")
        _, events = run_collect(self.root, "main.c")
        self.assertFalse(events[-1]["ok"])
        self.assertEqual(events[-1]["phase"], "compile")
        self.assertIn("undefined_thing", output_of(events))
        self.assertEqual([e["name"] for e in events if e["type"] == "phase"], ["compile"])


class ExternalScriptTests(unittest.TestCase):
    def test_script_contents_and_quoting(self):
        root = pathlib.Path(tempfile.mkdtemp())
        (root / "main.py").write_text("print(1)\n", encoding="utf-8")
        plan = runner.plan_run(root, "main.py", cfg())
        script = runner.external_script(plan, root)
        self.assertIn("cd /d ", script)
        self.assertIn("pause", script)
        self.assertIn('"' + sys.executable + '"', script)
        self.assertTrue(script.endswith("\r\n"))

    def test_rejects_dangerous_characters(self):
        with self.assertRaises(runner.BuildError):
            runner._bat_quote('a" & calc & "')
        with self.assertRaises(runner.BuildError):
            runner._bat_quote("%PATH%")
        self.assertEqual(runner._bat_quote("a&b.py"), '"a&b.py"')   # safe inside quotes
        self.assertEqual(runner._bat_quote("dir\\"), '"dir\\\\"')


if __name__ == "__main__":
    unittest.main()
