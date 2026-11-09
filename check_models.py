"""Find which Gemini models work with your API key. Run: python check_models.py"""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=15_000),
)

candidates = []
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    name = m.name.replace("models/", "")
    if "generateContent" in actions and "flash" in name and "image" not in name \
            and "tts" not in name and "live" not in name:
        candidates.append(name)

print(f"Testing {len(candidates)} flash models...\n")
for name in sorted(candidates):
    start = time.time()
    try:
        r = client.models.generate_content(model=name, contents="Say hello")
        print(f"OK    {name:45s} {time.time() - start:5.1f}s  -> {(r.text or '').strip()[:30]!r}")
    except Exception as e:
        print(f"FAIL  {name:45s} {time.time() - start:5.1f}s  {str(e)[:60]}")
