import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # progress bar is optional at runtime
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: str | Path) -> List[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def stable_id(*parts: str, prefix: str = "item") -> str:
    import hashlib

    raw = "||".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


_EXPLICIT_ANSWER_RE = re.compile(r"\b(?:answer|option|choice)\b[\s\S]{0,25}?\(?\b([ABCD])\b\)?", re.IGNORECASE)
_BARE_LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def option_letter(text: str) -> Optional[str]:
    """Extract a model's chosen MCQ option letter (A-D) from free-form text.

    Short, direct completions (the common case for an instruction-following
    model told to "return only the option letter") are handled by the bare
    fallback below. Long chain-of-thought responses need more care: naively
    case-folding the whole text and grabbing the *first* standalone A-D would
    also match the lowercase article "a" (folded to "A") or a sentence-initial
    "A ..." long before the model states its actual conclusion, and a reasoning
    trace often second-guesses itself ("the answer is B... wait, no, D") before
    landing on a final choice. So we (1) prefer the *last* explicit
    "answer/option/choice (is/might be/...) X" phrase, which is robust to
    self-correction, and (2) only fall back to the last bare, case-sensitive
    standalone letter -- never case-folded -- so the lowercase article isn't
    mistaken for option "A".
    """
    if not text:
        return None
    explicit = list(_EXPLICIT_ANSWER_RE.finditer(text))
    if explicit:
        return explicit[-1].group(1).upper()
    bare = list(_BARE_LETTER_RE.finditer(text))
    if bare:
        return bare[-1].group(1)
    return None


MANUAL_PREFIX = "manual:"
MANUAL_CACHE_DEFAULT = "runs/manual_llm_cache.json"
MANUAL_QUEUE_DEFAULT = "runs/manual_llm_queue.json"
MANUAL_PENDING = "[[PENDING_MANUAL_ANSWER]]"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class ChatClient:
    """Small OpenRouter-compatible client used by all generation tasks.

    Closed-source models are reached through the OpenRouter chat-completions API.
    To run local models, point ``base_url`` at any OpenAI-compatible local
    endpoint (e.g. vLLM, Ollama, llama.cpp server) or replace ``complete`` with a
    local inference wrapper. No API key is required when ``base_url`` is not
    the OpenRouter endpoint: local servers such as Ollama's OpenAI-compatible
    API (``http://localhost:11434/v1/chat/completions``) do not check one.

    A model name prefixed with ``manual:`` (e.g. ``manual:claude-sonnet-5``)
    switches to a key-free, file-backed "operator-in-the-loop" mode instead of
    calling a third-party API: ``complete`` looks up the exact message list in a
    local answer cache and returns it verbatim; on a cache miss it appends the
    rendered prompt (system+user messages only, no gold labels are ever part of
    these) to a queue file and returns a pending sentinel. An operator (human or
    another LLM instance) answers each queued prompt directly and writes the
    answers into the cache file keyed by the same hash, after which re-running
    the identical command reproduces the run exactly like an API-backed model
    would, through the same prompts/retrieval/scoring code. This is how the
    paper's Claude-evaluated rows were produced, with no OpenRouter key.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = 0.0,
        timeout: int = 120,
        manual_cache: str | Path = MANUAL_CACHE_DEFAULT,
        manual_queue: str | Path = MANUAL_QUEUE_DEFAULT,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.is_local = base_url != OPENROUTER_BASE_URL
        self.manual = model.startswith(MANUAL_PREFIX)
        if self.manual:
            self.manual_model = model[len(MANUAL_PREFIX) :]
            self.manual_cache_path = Path(manual_cache)
            self.manual_queue_path = Path(manual_queue)
            self._cache: Dict[str, str] = read_json(self.manual_cache_path, default={})
            self._queue: List[dict] = read_json(self.manual_queue_path, default=[])
            self._queue_keys = {row["key"] for row in self._queue}
            return
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key and not self.is_local:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY or pass api_key explicitly (or point --base-url "
                "at a local OpenAI-compatible server, which needs no key)."
            )

    def _manual_key(self, messages: List[Dict[str, str]]) -> str:
        return stable_id(self.manual_model, json.dumps(messages, sort_keys=True), prefix="manual")

    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 800) -> str:
        if self.manual:
            key = self._manual_key(messages)
            if key in self._cache:
                return self._cache[key]
            if key not in self._queue_keys:
                self._queue.append(
                    {"key": key, "model": self.manual_model, "messages": messages, "max_tokens": max_tokens}
                )
                self._queue_keys.add(key)
                write_json(self._queue, self.manual_queue_path)
            return MANUAL_PENDING

        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if not self.is_local:
            headers["HTTP-Referer"] = "https://github.com/lihuiliullh/AlzGraph"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if self.is_local:
            # Ollama-specific, harmless elsewhere: keep the model resident in
            # VRAM between calls instead of the ~5min default idle-unload, so a
            # multi-item task run doesn't pay a multi-minute reload per item.
            payload["keep_alive"] = "30m"
            # Thinking-capable local models (e.g. Qwen3) spend a chunk of the
            # token budget on an internal reasoning trace before the visible
            # answer; a task's per-call max_tokens (as low as 50 for a single
            # MCQ letter) is tuned for non-reasoning models and would cut the
            # response off mid-thought with no answer text at all. Give local
            # calls a generous floor instead of trusting the caller's budget.
            payload["max_tokens"] = max(max_tokens, 1536)
        for attempt in range(1, 4):
            response = requests.post(
                self.base_url, headers=headers, json=payload, timeout=self.timeout
            )
            if response.status_code == 429:
                time.sleep(min(30, 2**attempt))
                continue
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            content = (message.get("content") or "").strip()
            if not content and message.get("reasoning"):
                # The model was still inside its reasoning trace when it hit
                # the token budget (or the server otherwise never emitted a
                # final answer segment); fall back to the trace itself rather
                # than silently scoring an empty, always-wrong response.
                content = message["reasoning"].strip()
            return content
        raise RuntimeError(f"Chat completion request to {self.base_url} failed after retries.")


def batch(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
