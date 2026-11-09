"""Thin LLM wrapper. Swap providers by changing only this file."""
import os
import time

from google import genai
from google.genai import types

from .config import GEMINI_MODEL

_client = None
TIMEOUT_MS = 30_000  # give up on a single request after 30 s
RETRYABLE_CODES = {500, 503, 504}  # temporary server-side problems worth one retry


class LLMError(Exception):
    """An LLM failure with a message that is safe to show to the user."""


def _friendly(e: Exception) -> str:
    code = getattr(e, "code", None)
    if code == 429:
        return ("Gemini rate limit or free-tier quota reached. Wait a minute and try again, "
                "turn off 'Break down complex questions' to use fewer calls, or switch to "
                "another model with GEMINI_MODEL.")
    if code in (401, 403):
        return "Gemini rejected the API key. Check GEMINI_API_KEY and that it has access."
    if code == 404:
        return (f"Model '{GEMINI_MODEL}' isn't available for your key. "
                "Run `python check_models.py` and set GEMINI_MODEL to one that works.")
    if code == 400:
        return f"Gemini rejected the request (400). It may be an invalid API key. Details: {e}"
    if code in RETRYABLE_CODES or code is None:
        return "Gemini timed out or is overloaded right now. Please try again in a moment."
    return f"Gemini error ({code}): {e}"


def generate(prompt: str, temperature: float = 0.2) -> str:
    global _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set. Add it to .env or Streamlit secrets.")
    if _client is None:
        _client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS)
        )

    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        start = time.time()
        print(f"[llm] calling {GEMINI_MODEL} (attempt {attempt}/{max_attempts})", flush=True)
        try:
            resp = _client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config={"temperature": temperature}
            )
            print(f"[llm] ok in {time.time() - start:.1f}s", flush=True)
            return (resp.text or "").strip()
        except Exception as e:
            code = getattr(e, "code", None)
            print(f"[llm] failed after {time.time() - start:.1f}s: {e}", flush=True)
            retryable = code in RETRYABLE_CODES or code is None
            if attempt == max_attempts or not retryable:
                raise LLMError(_friendly(e)) from e
            time.sleep(2)
    raise LLMError("Unexpected LLM failure.")
