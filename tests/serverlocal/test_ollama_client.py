import json
from unittest.mock import MagicMock

import requests

from serverlocal import ollama_client


def test_stream_reply_yields_chunks_in_order(monkeypatch):
    lines = [
        json.dumps({"response": "Hola", "done": False}).encode(),
        json.dumps({"response": " mundo", "done": False}).encode(),
        json.dumps({"response": "", "done": True}).encode(),
    ]
    fake_response = MagicMock()
    fake_response.iter_lines.return_value = iter(lines)
    fake_response.raise_for_status.return_value = None
    captured = {}

    def fake_post(url, json, stream, timeout):
        captured["url"] = url
        captured["json"] = json
        return fake_response

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    chunks = list(ollama_client.stream_reply("hola"))

    assert chunks == ["Hola", " mundo"]
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "qwen2.5:7b"
    assert captured["json"]["stream"] is True
    fake_response.raise_for_status.assert_called_once()


def test_stream_reply_skips_blank_lines(monkeypatch):
    lines = [b"", json.dumps({"response": "hi", "done": True}).encode()]
    fake_response = MagicMock()
    fake_response.iter_lines.return_value = iter(lines)
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(ollama_client.requests, "post", lambda *a, **kw: fake_response)

    assert list(ollama_client.stream_reply("hi")) == ["hi"]


def test_stream_reply_raises_on_http_error(monkeypatch):
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = requests.HTTPError("503")
    monkeypatch.setattr(ollama_client.requests, "post", lambda *a, **kw: fake_response)

    try:
        list(ollama_client.stream_reply("hi"))
        assert False, "expected HTTPError"
    except requests.HTTPError:
        pass
