"""LLM access behind one small interface, chosen by LLM_PROVIDER / LLM_MODEL.

The LLM has two jobs only: turn a free-text request into a validated Intent, and word
an explanation from computed facts. It never sees tools it can call and never produces
numbers that are used. MOCK_LLM=true (or a missing key) disables it; callers then use
the rule-based parser and deterministic templates, so the whole pipeline still runs.
"""

import logging
import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)


class LLM:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model = os.getenv("LLM_MODEL", "gemini-3.8-flash")
        self.mock = os.getenv("MOCK_LLM", "false").lower() == "true"
        self._client = None
        self._lock = threading.Lock()
        self.last_error: str | None = None
        # circuit breaker: after a failed call, skip the LLM briefly so an outage costs
        # one slow request, not one per question (answers fall back to templates)
        self.cooldown_s = float(os.getenv("LLM_COOLDOWN_S", "60"))
        self._down_until = 0.0

    @property
    def enabled(self) -> bool:
        return not self.mock and self._has_key() and time.monotonic() >= self._down_until

    def _failed(self, e: Exception) -> None:
        self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
        self._down_until = time.monotonic() + self.cooldown_s
        log.warning("LLM call failed, using templates for %.0fs: %s", self.cooldown_s,
                    self.last_error)

    def _has_key(self) -> bool:
        keys = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY", "ollama": None}
        key = keys.get(self.provider, "missing")
        return key is None or bool(os.getenv(key or ""))

    def status(self) -> dict:
        return {"provider": self.provider, "model": self.model,
                "mode": ("live" if self.enabled else
                         "cooling_down" if (not self.mock and self._has_key()) else "mock"),
                "last_error": self.last_error}

    def client(self):
        """Build the chat model lazily. Only the Gemini package is in requirements.txt;
        the other providers work if their langchain package is installed."""
        with self._lock:
            if self._client is not None:
                return self._client
            common = {"temperature": 0, "max_retries": 1}
            if self.provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._client = ChatGoogleGenerativeAI(
                    model=self.model, google_api_key=os.environ["GEMINI_API_KEY"],
                    timeout=12, **common)
            elif self.provider == "openai":
                from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
                self._client = ChatOpenAI(model=self.model, timeout=30, **common)
            elif self.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
                self._client = ChatAnthropic(model=self.model, timeout=30, **common)
            elif self.provider == "ollama":
                from langchain_ollama import ChatOllama  # type: ignore[import-not-found]
                self._client = ChatOllama(
                    model=self.model, temperature=0,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
            else:
                raise ValueError(f"unknown LLM_PROVIDER {self.provider}")
            return self._client

    def structured(self, schema, messages) -> tuple[object | None, dict]:
        """Return (parsed schema instance or None, usage). Never raises."""
        if not self.enabled:
            return None, _usage()
        start = time.perf_counter()
        try:
            out = self.client().with_structured_output(schema, include_raw=True).invoke(messages)
            self.last_error = None
            return out.get("parsed"), _usage(out.get("raw"), start)
        except Exception as e:  # network, quota, bad JSON: fall back, never crash
            self._failed(e)
            return None, _usage(None, start, failed=True)

    def text(self, messages) -> tuple[str | None, dict]:
        """Return (text or None, usage). Never raises."""
        if not self.enabled:
            return None, _usage()
        start = time.perf_counter()
        try:
            msg = self.client().invoke(messages)
            self.last_error = None
            content = msg.content if isinstance(msg.content, str) else "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in msg.content)
            return content.strip(), _usage(msg, start)
        except Exception as e:
            self._failed(e)
            return None, _usage(None, start, failed=True)


def _usage(msg=None, start: float | None = None, failed: bool = False) -> dict:
    meta = getattr(msg, "usage_metadata", None) or {}
    called = start is not None
    return {"calls": 1 if called else 0, "failed": failed,
            "input_tokens": int(meta.get("input_tokens", 0)),
            "output_tokens": int(meta.get("output_tokens", 0)),
            "total_tokens": int(meta.get("total_tokens", 0)),
            "ms": round((time.perf_counter() - start) * 1000, 1) if called else 0.0}


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


def reset_llm() -> None:
    """Re-read the environment (tests and config changes)."""
    global _llm
    _llm = None
