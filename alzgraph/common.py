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


def option_letter(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


MANUAL_PREFIX = "manual:"
MANUAL_CACHE_DEFAULT = "runs/manual_llm_cache.json"
MANUAL_QUEUE_DEFAULT = "runs/manual_llm_queue.json"
MANUAL_PENDING = "[[PENDING_MANUAL_ANSWER]]"


class ChatClient:
    """Small OpenRouter-compatible client used by all generation tasks.

    Closed-source models are reached through the OpenRouter chat-completions API.
    To run local models, point ``base_url`` at any OpenAI-compatible local
    endpoint (e.g. vLLM, Ollama, llama.cpp server) or replace ``complete`` with a
    local inference wrapper.

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
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        temperature: float = 0.0,
        timeout: int = 120,
        manual_cache: str | Path = MANUAL_CACHE_DEFAULT,
        manual_queue: str | Path = MANUAL_QUEUE_DEFAULT,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
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
        if not self.api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY or pass api_key explicitly.")

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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/lihuiliullh/AlzGraph",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(1, 4):
            response = requests.post(
                self.base_url, headers=headers, json=payload, timeout=self.timeout
            )
            if response.status_code == 429:
                time.sleep(min(30, 2**attempt))
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError("OpenRouter request failed after retries.")


def batch(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    chunk: List[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
