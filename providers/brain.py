"""A small, transparent NEXUS reference matcher with built-in and owner examples.

This is intentionally not a pretrained language model. It builds a TF-IDF
question matcher from scratch and returns the answer attached to the closest
reference question. When it does not know, it says so instead of fabricating an
answer. A lightweight arithmetic rule and project-text retrieval are included.
"""

from __future__ import annotations

import ast
import json
import math
import os
import pathlib
import re
import statistics
import time
import unicodedata
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from providers import ChatResult
from providers.brain_knowledge import BUILTIN_EXAMPLES


class NexusBrainProvider:
    provider_id = "nexus-brain"
    display_name = "NEXUS Brain Lite"
    model_name = "nexus-brain-lite"
    supports_edits = False
    _stop = frozenset({
        "a", "an", "and", "are", "as", "at", "be", "can", "could", "do", "does",
        "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
        "please", "should", "tell", "that", "the", "this", "to", "was", "what", "when",
        "where", "which", "who", "why", "with", "would", "you", "your",
    })
    _greetings = frozenset({"hello", "hi", "hey", "greetings"})
    _max_examples = 5000
    _max_question_chars = 3000
    _max_answer_chars = 12000

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_path = self._data_dir() / "brain.json"
        self.seed_examples = [dict(item, aliases=list(item.get("aliases", [])), built_in=True)
                              for item in BUILTIN_EXAMPLES]
        self._user_examples: List[dict] = []
        self.examples: List[dict] = []
        self._documents: List[dict] = []
        self._idf: Dict[str, float] = {}
        self._load()

    @staticmethod
    def _data_dir() -> pathlib.Path:
        configured = os.environ.get("NEXUS_BRAIN_DIR")
        if configured:
            return pathlib.Path(configured).expanduser()
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return pathlib.Path(local) / "Nexus" / "brain"
        return pathlib.Path.home() / ".local" / "share" / "nexus" / "brain"

    @classmethod
    def _terms(cls, text: str) -> List[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        terms = []
        words = []
        current = []
        for char in normalized:
            category = unicodedata.category(char)
            if category[0] in ("L", "N", "M") or char in "_#+.-":
                current.append(char)
            elif current:
                words.append("".join(current))
                current.clear()
        if current:
            words.append("".join(current))

        for word in words:
            if len(word) > 1 and word not in cls._stop:
                terms.append(word)
            # CJK, Thai, and Lao often omit spaces between words. Character
            # bigrams give the local matcher useful overlap without a language
            # dictionary or a third-party tokenizer.
            if any((0x3400 <= ord(char) <= 0x9FFF or 0x3040 <= ord(char) <= 0x30FF
                    or 0x0E00 <= ord(char) <= 0x0E7F or 0x0E80 <= ord(char) <= 0x0EFF)
                   for char in word):
                terms.extend("\u241f".join(word[index:index + 2])
                             for index in range(max(0, len(word) - 1)))
        return terms

    @staticmethod
    def _normalized_question(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    def _load(self) -> None:
        try:
            data = json.loads(self.data_path.read_text(encoding="utf-8"))
            examples = data.get("examples", []) if isinstance(data, dict) else []
            if isinstance(examples, list):
                self._user_examples = [
                    {"question": str(item.get("question", "")).strip(),
                     "answer": str(item.get("answer", "")).strip(),
                     "trained_at": float(item.get("trained_at", 0)),
                     "category": "My examples", "aliases": []}
                    for item in examples
                    if isinstance(item, dict) and str(item.get("question", "")).strip()
                    and str(item.get("answer", "")).strip()
                ][-self._max_examples:]
        except (OSError, ValueError, TypeError):
            self._user_examples = []
        self._rebuild_examples()

    def _rebuild_examples(self) -> None:
        """Combine built-in references with owner examples; owner answers can override a seed."""
        self.examples = [dict(item, aliases=list(item.get("aliases", [])))
                         for item in self.seed_examples]
        seed_lookup = {}
        for index, example in enumerate(self.examples):
            for phrase in [example["question"], *example.get("aliases", [])]:
                seed_lookup[self._normalized_question(phrase)] = index
        for custom in self._user_examples:
            key = self._normalized_question(custom["question"])
            seed_index = seed_lookup.get(key)
            if seed_index is not None:
                seed = self.examples[seed_index]
                seed["answer"] = custom["answer"]
                seed["customized"] = True
                if key != self._normalized_question(seed["question"]):
                    seed.setdefault("aliases", []).append(custom["question"])
            else:
                self.examples.append(dict(custom, built_in=False))
        self._fit()

    def _fit(self) -> None:
        self._documents = []
        frequency = Counter()
        for index, example in enumerate(self.examples):
            phrases = [example["question"], *example.get("aliases", [])]
            for phrase in phrases:
                terms = Counter(self._terms(phrase))
                if not terms:
                    continue
                self._documents.append({"index": index, "terms": terms, "norm": 0.0})
                frequency.update(terms.keys())
        count = max(1, len(self._documents))
        self._idf = {term: math.log(1.0 + count / (1.0 + seen)) + 1.0
                     for term, seen in frequency.items()}
        for document in self._documents:
            document["norm"] = math.sqrt(sum((amount * self._idf.get(term, 1.0)) ** 2
                                              for term, amount in document["terms"].items()))

    def _save(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.data_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 2, "examples": self._user_examples},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.data_path)

    def train_pair(self, question: str, answer: str) -> dict:
        question = str(question).strip()
        answer = str(answer).strip()
        if not question or not answer:
            raise ValueError("Enter both a question and its preferred answer.")
        if len(question) > self._max_question_chars or len(answer) > self._max_answer_chars:
            raise ValueError("The question or answer is too long.")
        key = self._normalized_question(question)
        replacement = {"question": question, "answer": answer, "trained_at": time.time(),
                       "category": "My examples", "aliases": []}
        existing = next((i for i, item in enumerate(self._user_examples)
                         if self._normalized_question(item["question"]) == key), None)
        if existing is None:
            if len(self._user_examples) >= self._max_examples:
                raise ValueError("The brain has reached its 5,000 example limit.")
            self._user_examples.append(replacement)
        else:
            self._user_examples[existing] = replacement
        self._rebuild_examples()
        self._save()
        return self.status()

    def list_models(self) -> List[str]:
        return [self.model_name]

    def ensure_model(self) -> None:
        if str(self.cfg.get("model", self.model_name)) != self.model_name:
            raise RuntimeError(f"NEXUS Brain only provides '{self.model_name}'.")

    def status(self) -> dict:
        installed = str(self.cfg.get("model", self.model_name)) == self.model_name
        categories = Counter(item["category"] for item in self.seed_examples)
        return {"provider_id": self.provider_id, "provider_name": self.display_name,
                "ok": True, "models": self.list_models(), "installed": installed, "error": "",
                "trained_examples": len(self._user_examples),
                "built_in_examples": len(self.seed_examples),
                "available_examples": len(self.examples),
                "knowledge_categories": dict(sorted(categories.items())),
                "knowledge_topics": [{"category": item["category"], "title": item["title"]}
                                     for item in self.seed_examples],
                "supports_edits": False}

    def _match(self, prompt: str) -> Tuple[Optional[dict], float]:
        terms = Counter(self._terms(prompt))
        if not terms:
            return None, 0.0
        query_norm = math.sqrt(sum((amount * self._idf.get(term, 1.0)) ** 2
                                   for term, amount in terms.items()))
        if not query_norm:
            return None, 0.0
        best_index, best_score = None, 0.0
        for index, document in enumerate(self._documents):
            dot = sum((amount * self._idf.get(term, 1.0)) *
                      (document["terms"].get(term, 0) * self._idf.get(term, 1.0))
                      for term, amount in terms.items())
            denominator = query_norm * document["norm"]
            score = dot / denominator if denominator else 0.0
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            return None, best_score
        return self.examples[self._documents[best_index]["index"]], best_score

    @staticmethod
    def _format_number(value: float) -> str:
        if abs(value - round(value)) < 1e-10:
            return str(int(round(value)))
        return f"{value:.10g}"

    @staticmethod
    def _solve_equation(prompt: str) -> Optional[str]:
        """Solve a single real linear or quadratic equation in x using a safe AST."""
        if "=" not in prompt or not re.search(r"(?i)\bx\b", prompt):
            return None
        equation_text = prompt.replace("−", "-").replace("×", "*")
        source = re.sub(r"(?i)^\s*(?:(?:please\s+)?solve(?:\s+for\s+x)?(?:\s+equation)?|find\s+x(?=\s*:)|what\s+is)\s*:?\s*",
                        "", equation_text).strip().rstrip("?. ")
        if source.count("=") != 1 or not re.fullmatch(r"[0-9xX.+*/()=\-^\s]+", source):
            return None
        source = re.sub(r"(?i)(\d|\))\s*x\b", r"\1*x", source)
        source = re.sub(r"(?i)\bx\s*\(", "x*(", source)
        source = re.sub(r"(?<=\d)\s*\(", "*(", source)
        source = re.sub(r"(?<=\))\s*\(", ")*(", source).replace("^", "**")
        left_text, right_text = source.split("=", 1)
        try:
            left_tree = ast.parse(left_text, mode="eval")
            right_tree = ast.parse(right_text, mode="eval")
        except (SyntaxError, ValueError):
            return None

        def polynomial(node):
            if isinstance(node, ast.Expression):
                return polynomial(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return [float(node.value), 0.0, 0.0]
            if isinstance(node, ast.Name) and node.id.casefold() == "x":
                return [0.0, 1.0, 0.0]
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = polynomial(node.operand)
                return value if isinstance(node.op, ast.UAdd) else [-part for part in value]
            if isinstance(node, ast.BinOp):
                left, right = polynomial(node.left), polynomial(node.right)
                if isinstance(node.op, ast.Add):
                    result = [a + b for a, b in zip(left, right)]
                elif isinstance(node.op, ast.Sub):
                    result = [a - b for a, b in zip(left, right)]
                elif isinstance(node.op, ast.Mult):
                    result = [0.0, 0.0, 0.0]
                    for i, a in enumerate(left):
                        for j, b in enumerate(right):
                            if i + j > 2 and a and b:
                                raise ValueError("degree greater than two")
                            if i + j <= 2:
                                result[i + j] += a * b
                elif isinstance(node.op, ast.Div) and right[1] == 0 and right[2] == 0 and right[0] != 0:
                    result = [a / right[0] for a in left]
                elif isinstance(node.op, ast.Pow) and right[1] == 0 and right[2] == 0 \
                        and right[0] in (0, 1, 2):
                    power = int(right[0])
                    result = [1.0, 0.0, 0.0]
                    if power == 1:
                        result = left
                    elif power == 2:
                        result = [left[0] * left[0], 2 * left[0] * left[1],
                                  left[1] * left[1] + 2 * left[0] * left[2]]
                        if left[2] and (left[1] or left[2]):
                            raise ValueError("degree greater than two")
                else:
                    raise ValueError("unsupported polynomial operation")
                if any(not math.isfinite(value) or abs(value) > 1e12 for value in result):
                    raise ValueError("coefficient out of range")
                return result
            raise ValueError("unsupported equation")

        try:
            lhs, rhs = polynomial(left_tree), polynomial(right_tree)
            c, b, a = [x - y for x, y in zip(lhs, rhs)]
        except (ValueError, TypeError, OverflowError, RecursionError):
            return None

        epsilon = 1e-12
        if abs(a) < epsilon:
            if abs(b) < epsilon:
                return "Every value of x satisfies this equation." if abs(c) < epsilon \
                    else "This equation has no solution."
            root = -c / b
            return (f"Linear equation: {source}\n"
                    f"Rearrange to {NexusBrainProvider._format_number(b)}x = "
                    f"{NexusBrainProvider._format_number(-c)}.\n"
                    f"Solution: x = {NexusBrainProvider._format_number(root)}")

        discriminant = b * b - 4 * a * c
        pieces = []
        for coefficient, label in ((a, "x²"), (b, "x"), (c, "")):
            if abs(coefficient) < epsilon:
                continue
            magnitude = abs(coefficient)
            shown = "" if label and abs(magnitude - 1.0) < epsilon else NexusBrainProvider._format_number(magnitude)
            term = shown + label
            if not pieces:
                pieces.append(("-" if coefficient < 0 else "") + term)
            else:
                pieces.append((" - " if coefficient < 0 else " + ") + term)
        standard = "".join(pieces) + " = 0"
        if discriminant >= -epsilon:
            discriminant = max(0.0, discriminant)
            if discriminant == 0:
                numeric_roots = [-b / (2 * a)]
            else:
                q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
                numeric_roots = [q / a, c / q] if q else [-b / (2 * a)]
            numeric_roots.sort()
            roots = [NexusBrainProvider._format_number(root) for root in numeric_roots]
            result = " and ".join("x = " + value for value in roots)
            return (f"Standard form: {standard}\n"
                    f"Discriminant: D = b² − 4ac = {NexusBrainProvider._format_number(discriminant)}.\n"
                    f"Using x = (−b ± √D) / (2a): {result}")
        real_part = -b / (2 * a)
        imaginary_part = math.sqrt(-discriminant) / abs(2 * a)
        return (f"Standard form: {standard}\n"
                f"Discriminant: D = {NexusBrainProvider._format_number(discriminant)}, so the roots are complex.\n"
                f"Solutions: x = {NexusBrainProvider._format_number(real_part)} ± "
                f"{NexusBrainProvider._format_number(imaginary_part)}i")

    @staticmethod
    def _safe_arithmetic(prompt: str) -> Optional[str]:
        """Evaluate bounded arithmetic expressions without eval or arbitrary Python."""
        source = prompt.strip().replace("×", "*").replace("÷", "/").replace("−", "-")
        source = re.sub(r"(?i)^\s*(?:(?:please\s+)?(?:calculate|compute|evaluate|work\s+out|what\s+is|what's|how\s+much\s+is)(?:\s+the)?)\s+",
                        "", source).strip().rstrip("?. ")
        source = re.sub(r"(?i)(?:square\s+root|sqrt)\s+of\s+(\d+(?:\.\d+)?)", r"sqrt(\1)", source)
        source = re.sub(r"(?i)factorial\s+of\s+(\d+)", r"factorial(\1)", source)
        source = re.sub(r"(?i)(?:gcd|greatest\s+common\s+divisor)\s+of\s+(\d+)\s+and\s+(\d+)", r"gcd(\1,\2)", source)
        source = re.sub(r"(?i)(?:lcm|least\s+common\s+multiple)\s+of\s+(\d+)\s+and\s+(\d+)", r"lcm(\1,\2)", source)
        source = re.sub(r"(?i)(\d+)\s+to\s+the\s+power\s+of\s+(\d+)", r"\1**\2", source)
        source = re.sub(r"(?<![\w.])(\d+)\s*!", r"factorial(\1)", source)
        source = re.sub(r"(?i)(mean|average|median|stdev|pstdev|variance|pvariance|sum|min|max)\s+of\s+([\d.,\s+\-and]+)",
                        lambda match: ("mean" if match.group(1).casefold() == "average" else match.group(1).casefold())
                        + "(" + re.sub(r"(?i)\band\b", ",", match.group(2)).strip(" ,") + ")", source)
        source = re.sub(r"(?i)what\s+percent\s+is\s+(\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)",
                        r"(\1/\2)*100", source)
        source = re.sub(r"(?i)increase\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s*(?:%|percent)", r"(\1)*(1+\2/100)", source)
        source = re.sub(r"(?i)decrease\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s*(?:%|percent)", r"(\1)*(1-\2/100)", source)
        source = re.sub(r"(?i)(?:percentage\s+)?change\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)",
                        r"((\2-\1)/\1)*100", source)
        source = re.sub(r"(?i)(\d+(?:\.\d+)?)\s+of\s+(\d+(?:\.\d+)?)", r"(\1)*(\2)", source)
        source = re.sub(r"(?i)(\d+)\s+choose\s+(\d+)", r"comb(\1,\2)", source)
        source = re.sub(r"(?i)\bdivided\s+by\b", "/", source)
        source = re.sub(r"(?i)\bmultiplied\s+by\b|\btimes\b", "*", source)
        source = re.sub(r"(?i)\bplus\b", "+", source)
        source = re.sub(r"(?i)\bminus\b", "-", source)
        source = re.sub(r"(?i)\bmod(?:ulo)?\b", "%", source)
        source = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)",
                        r"(\1/100)*(\2)", source)
        source = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*percent\b", r"(\1/100)", source)
        source = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%(?!\s*\d)", r"(\1/100)", source)
        source = re.sub(r"(?<=\d)\s*\(", "*(", source)
        source = re.sub(r"(?<=\))\s*\(", ")*(", source)
        source = re.sub(r"(?<=\))\s*(?=\d)", "*", source)
        source = re.sub(r"(?i)(\d)\s*(pi|tau)\b", r"\1*\2", source)
        source = source.replace("^", "**")
        if len(source) > 300 or not re.fullmatch(r"[0-9A-Za-z_+*/%().,!\-\s]+", source):
            return None
        try:
            tree = ast.parse(source, mode="eval")
        except (SyntaxError, ValueError):
            return None
        if sum(1 for _ in ast.walk(tree)) > 100:
            return None

        functions = {
            "abs": abs, "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
            "factorial": math.factorial, "gcd": math.gcd, "lcm": math.lcm,
            "comb": math.comb, "perm": math.perm, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "log": math.log, "log10": math.log10, "exp": math.exp, "round": round,
            "degrees": math.degrees, "radians": math.radians, "hypot": math.hypot,
            "min": min, "max": max, "sum": lambda *values: sum(values),
            "mean": lambda *values: statistics.mean(values),
            "median": lambda *values: statistics.median(values),
            "stdev": lambda *values: statistics.stdev(values),
            "pstdev": lambda *values: statistics.pstdev(values),
            "variance": lambda *values: statistics.variance(values),
            "pvariance": lambda *values: statistics.pvariance(values),
        }

        def checked(value):
            if isinstance(value, complex) or not isinstance(value, (int, float)):
                raise ValueError("only real numeric results are supported")
            if not math.isfinite(float(value)) or abs(value) > 1e15:
                raise ValueError("result is outside the supported range")
            return value

        def calculate(node):
            if isinstance(node, ast.Expression):
                return calculate(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return checked(node.value)
            if isinstance(node, ast.Name) and node.id.casefold() in {"pi", "e", "tau"}:
                return {"pi": math.pi, "e": math.e, "tau": math.tau}[node.id.casefold()]
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = calculate(node.operand)
                return checked(value if isinstance(node.op, ast.UAdd) else -value)
            if isinstance(node, ast.BinOp):
                left, right = calculate(node.left), calculate(node.right)
                if isinstance(node.op, ast.Add): result = left + right
                elif isinstance(node.op, ast.Sub): result = left - right
                elif isinstance(node.op, ast.Mult): result = left * right
                elif isinstance(node.op, ast.Div): result = left / right
                elif isinstance(node.op, ast.FloorDiv): result = left // right
                elif isinstance(node.op, ast.Mod): result = left % right
                elif isinstance(node.op, ast.Pow):
                    if abs(right) > 100 or abs(left) > 1e6:
                        raise ValueError("power is outside the supported range")
                    result = left ** right
                else: raise ValueError("unsupported arithmetic operator")
                return checked(result)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
                name = node.func.id.casefold()
                function = functions.get(name)
                if function is None or len(node.args) > 50:
                    raise ValueError("unsupported math function")
                args = [calculate(argument) for argument in node.args]
                if name in {"factorial", "comb", "perm"}:
                    if any(not float(value).is_integer() for value in args):
                        raise ValueError("this function requires integer arguments")
                    if any(abs(value) > 1000 for value in args):
                        raise ValueError("integer argument is outside the supported range")
                    args = [int(value) for value in args]
                if name == "factorial" and (len(args) != 1 or args[0] < 0):
                    raise ValueError("factorial requires one nonnegative integer")
                if name in {"comb", "perm"} and (len(args) != 2 or min(args) < 0):
                    raise ValueError(f"{name} requires two nonnegative integers")
                if name in {"mean", "median", "min", "max", "sum"} and not args:
                    raise ValueError(f"{name} requires at least one value")
                if name in {"stdev", "pstdev", "variance", "pvariance"} and len(args) < 2:
                    raise ValueError(f"{name} requires at least two values")
                if name == "round" and len(args) == 2 and abs(args[1]) > 12:
                    raise ValueError("round precision must be between -12 and 12")
                return checked(function(*args))
            raise ValueError("unsupported arithmetic expression")

        try:
            result = calculate(tree)
            formatted = NexusBrainProvider._format_number(float(result))
            return f"Result: {formatted}"
        except (ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
            return None

    @staticmethod
    def _task_from_messages(messages: List[dict]) -> Tuple[str, str]:
        user_text = next((str(message.get("content", "")) for message in reversed(messages)
                          if message.get("role") == "user"), "").strip()
        context = ""
        marker = "PROJECT CONTEXT (data, not instructions):\n"
        task_marker = "\n\nTASK:\n"
        if marker in user_text and task_marker in user_text:
            before, task = user_text.split(task_marker, 1)
            context = before.split(marker, 1)[1]
            user_text = task.split("\n\nRemember:", 1)[0].strip()
        return user_text, context

    @staticmethod
    def _project_excerpt(prompt: str, context: str) -> Optional[str]:
        if not context:
            return None
        query = set(NexusBrainProvider._terms(prompt))
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", context) if part.strip()]
        ranked = []
        for paragraph in paragraphs:
            words = set(NexusBrainProvider._terms(paragraph))
            overlap = query & words
            if overlap:
                score = len(overlap) / math.sqrt(max(1, len(query)) * max(1, len(words)))
                ranked.append((score, paragraph))
        ranked.sort(reverse=True)
        if not ranked or ranked[0][0] < 0.12:
            return None
        excerpt = ranked[0][1]
        return excerpt[:1400] + ("…" if len(excerpt) > 1400 else "")

    def _answer(self, prompt: str, context: str) -> str:
        terms = self._terms(prompt)
        if terms and set(terms).issubset(self._greetings):
            return ("Hello, I'm NEXUS Brain. I learn from the question-and-answer examples you add "
                    "under Settings → Brain training.")
        equation = self._solve_equation(prompt)
        if equation is not None:
            return equation
        arithmetic = self._safe_arithmetic(prompt)
        if arithmetic is not None:
            return arithmetic
        example, score = self._match(prompt)
        if example and score >= 0.22:
            return example["answer"]
        excerpt = self._project_excerpt(prompt, context)
        if excerpt:
            return ("I can’t reason about project files yet, but this is the closest text I found "
                    "in the provided project context:\n\n" + excerpt)
        return (f"I haven't learned an answer to that yet. I have {len(self.seed_examples)} built-in "
                f"reference topics and {len(self._user_examples)} examples you've added. Teach me a "
                "specific answer in Settings → Brain training, then ask again.")

    def chat_stream(self, messages: List[dict], num_ctx: int, num_predict: int,
                    on_piece: Callable[[str], None]) -> ChatResult:
        started = time.perf_counter()
        prompt, context = self._task_from_messages(messages)
        answer = self._answer(prompt, context)
        on_piece(answer)
        elapsed_ns = int((time.perf_counter() - started) * 1e9)
        return ChatResult(answer, {"model": self.model_name,
                                   "prompt_tokens": len(self._terms(prompt)),
                                   "completion_tokens": len(self._terms(answer)),
                                   "completion_duration_ns": max(1, elapsed_ns),
                                   "finish_reason": "stop"})
