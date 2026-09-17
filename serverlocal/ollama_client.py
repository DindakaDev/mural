import json
from collections.abc import Iterator

import requests

OLLAMA_HOST = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:7b"


def stream_reply(prompt: str, model: str = CHAT_MODEL) -> Iterator[str]:
    """Yields text chunks as Ollama generates them. Raises
    requests.HTTPError on a non-2xx response, or requests.ConnectionError
    if Ollama isn't reachable."""
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        payload = json.loads(line)
        chunk = payload.get("response")
        if chunk:
            yield chunk
        if payload.get("done"):
            break
