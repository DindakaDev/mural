# Local Voice Server — Barge-in, Delegation & Lessons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add barge-in (real interruption), the client↔server delegation
protocol (research handoff mid-conversation), full data-channel message
handling (mute/close/context injection), and a local `/responses` endpoint
(JSON-schema + one web-search tool call) to the local voice server built in
the companion "core pipeline" plan — closing the remaining gaps between
this server and what the Android app's protocol actually expects.

**Architecture:** Two small, independently-testable additions to the
existing turn-detection/response pipeline (speech-run tracking for
barge-in, a token-stream filter for delegation markers) sit alongside two
new standalone modules (a DuckDuckGo search tool, an Ollama chat/tool-call
wrapper) behind a new `/responses` route. `session.py` — already the
integration point for the core pipeline — gains a real data-channel
message handler, task-tracked cancellable turns, and minimal connection
lifecycle cleanup.

**Tech Stack:** Same as the core pipeline plan (`fastapi`, `aiortc`,
`webrtcvad`, `faster-whisper`, `kokoro-onnx`, `requests`), plus
`duckduckgo-search` for the web-search tool.

**Spec:** `docs/superpowers/specs/2026-09-15-local-voice-server-design.md`

## Global Constraints

- Barge-in threshold: sustained speech ≥200ms while the assistant is
  speaking cancels the in-flight response and clears queued audio. (Spec:
  "Barge-in".)
- The delegation marker convention (`[[SEARCH: <query>]]`) is this
  server's own choice, not part of the wire protocol — only
  `session.delegation.created` (server→client) and
  `session.commentary.append` (client→server) cross the wire, matching
  `MuralViewModel.kt:753-756` (`if (d["target"]?.jsonPrimitive?.content ==
  "client") d["id"]?.jsonPrimitive?.content?.let(::delegate)`) and
  `MuralViewModel.kt:897` (`command("commentary", result.text, id)`).
- `/responses`' request/response shape must satisfy
  `TeachingResponse.kt:8-45`'s `decodeTeachingResponse` EXACTLY:
  `status` must be the literal string `"completed"`; at least one
  `output[].content[].type == "output_text"` item with non-empty `text`;
  a `type == "refusal"` content item is treated as a refusal by the
  client; `output[].type == "web_search_call"` items are counted as
  search calls; citations need `content[].annotations[].type ==
  "url_citation"` with an `https` URL, non-blank host, no userinfo (else
  the client silently drops them — see `isSafeSourceUrl`); usage comes
  from `usage.input_tokens`/`usage.output_tokens`.
- `max_tool_calls` is always `1` when the Android client requests search
  (`APIClient.kt`'s `respond()`: `put("max_tool_calls", 1)`) — the
  server-side tool loop must cap at exactly one search call, no retries.
- Client `input` items for `/responses` have a **plain string** `content`
  field (`APIClient.kt`'s `respond()`: `put("content", input)`), NOT the
  typed content-array shape `ConversationHistory.messages()` builds for
  live-session history — do not assume the array form here.
- Existing files this plan builds on (from the core pipeline plan, already
  merged): `serverlocal/vad.py` (`TurnDetector`), `serverlocal/
  input_pipeline.py` (`InputPipeline`), `serverlocal/session.py`
  (`Session`, `build_input_pipeline`, `run_response_turn`,
  `create_live_session`, `_consume_audio`), `serverlocal/
  response_pipeline.py` (`ResponsePipeline`, reused as-is for commentary
  playback), `serverlocal/events.py`, `serverlocal/ollama_client.py`
  (`stream_reply`), `serverlocal/models.py` (`synthesize`).

---

## File Structure

```
serverlocal/
  vad.py                  # modify — TurnDetector gains speech_run_ms tracking
  input_pipeline.py        # modify — InputPipeline.current_speech_run_ms()
  events.py                 # modify — add delegation_created()
  delegation.py              # new — strip_delegation_markers()
  search_tool.py              # new — web_search() via duckduckgo-search
  ollama_responses.py          # new — generate(): Ollama chat + tool-calling + schema
  responses_api.py              # new — FastAPI POST /responses route
  server.py                      # modify — include_router(responses_api.router)
  session.py                      # modify — barge-in, delegation wiring,
                                   #   data-channel message handler, session
                                   #   lifecycle cleanup
  requirements.txt                 # modify — add duckduckgo-search
tests/serverlocal/
  test_vad.py               # modify — speech_run_ms tests
  test_input_pipeline.py     # modify — current_speech_run_ms() tests
  test_events.py              # modify — delegation_created() test
  test_delegation.py           # new
  test_search_tool.py           # new
  test_ollama_responses.py       # new
  test_responses_api.py           # new
  test_session.py                  # modify — barge-in, delegation, message
                                    #   handler, close/cleanup tests
```

---

### Task 1: Barge-in signal — `TurnDetector.speech_run_ms` + `InputPipeline.current_speech_run_ms()`

**Files:**
- Modify: `serverlocal/vad.py` (the whole `TurnDetector` class)
- Modify: `tests/serverlocal/test_vad.py`
- Modify: `serverlocal/input_pipeline.py` (the whole `InputPipeline` class)
- Modify: `tests/serverlocal/test_input_pipeline.py`

**Interfaces:**
- Produces: `TurnDetector.speech_run_ms: int` (public attribute, ms of
  *consecutive* speech frames right now — resets to 0 on any non-speech
  frame, independent of turn open/close state), `InputPipeline
  .current_speech_run_ms() -> int`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/serverlocal/test_vad.py` (after the existing tests):

```python
def test_speech_run_ms_accumulates_during_continuous_speech_and_resets_on_silence(monkeypatch):
    pattern = [True, True, True, False, True, True]
    calls = iter(pattern)
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: next(calls))

    detector = TurnDetector()
    observed = []
    for _ in range(len(pattern)):
        detector.feed(_frame_bytes())
        observed.append(detector.speech_run_ms)

    assert observed == [20, 40, 60, 0, 20, 40]


def test_speech_run_ms_stays_zero_during_pure_silence(monkeypatch):
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: False)

    detector = TurnDetector()
    for _ in range(10):
        detector.feed(_frame_bytes())

    assert detector.speech_run_ms == 0
```

Add to `tests/serverlocal/test_input_pipeline.py` (after the existing tests):

```python
def test_current_speech_run_ms_reflects_ongoing_speech(monkeypatch):
    pattern = [True, True, False]
    calls = iter(pattern)
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: next(calls))

    pipeline = InputPipeline(lambda a: "x", lambda *a: None)
    observed = []
    for _ in range(len(pattern)):
        pipeline.handle_frame(_silence_frame())
        observed.append(pipeline.current_speech_run_ms())

    assert observed == [20, 40, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_vad.py tests/serverlocal/test_input_pipeline.py -v`
Expected: FAIL — `AttributeError: 'TurnDetector' object has no attribute
'speech_run_ms'` and `AttributeError: 'InputPipeline' object has no
attribute 'current_speech_run_ms'`.

- [ ] **Step 3: Update `serverlocal/vad.py`**

```python
import webrtcvad


class TurnDetector:
    """Buffers 16kHz mono 16-bit PCM, classifies 20ms frames with WebRTC
    VAD, and reports a closed turn once >=500ms of trailing silence
    follows speech. Also tracks speech_run_ms -- consecutive ms of speech
    happening right now, independent of turn state -- for barge-in."""

    SAMPLE_RATE = 16000
    FRAME_MS = 20
    FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # int16 = 2 bytes/sample
    SILENCE_MS_TO_CLOSE = 500

    def __init__(self, vad_mode: int = 2):
        self._vad = webrtcvad.Vad(vad_mode)
        self._buffer = bytearray()
        self._turn_audio = bytearray()
        self._in_speech = False
        self._silence_ms = 0
        self.speech_run_ms = 0

    def feed(self, pcm_bytes: bytes) -> bytes | None:
        self._buffer += pcm_bytes
        closed_turn = None
        while len(self._buffer) >= self.FRAME_BYTES:
            frame = bytes(self._buffer[: self.FRAME_BYTES])
            del self._buffer[: self.FRAME_BYTES]
            is_speech = self._vad.is_speech(frame, self.SAMPLE_RATE)
            if is_speech:
                self.speech_run_ms += self.FRAME_MS
                self._in_speech = True
                self._silence_ms = 0
                self._turn_audio += frame
            else:
                self.speech_run_ms = 0
                if self._in_speech:
                    self._turn_audio += frame
                    self._silence_ms += self.FRAME_MS
                    if self._silence_ms >= self.SILENCE_MS_TO_CLOSE:
                        closed_turn = bytes(self._turn_audio)
                        self._turn_audio = bytearray()
                        self._in_speech = False
                        self._silence_ms = 0
        return closed_turn
```

- [ ] **Step 4: Update `serverlocal/input_pipeline.py`**

```python
from collections.abc import Callable

import av
import numpy as np

from .vad import TurnDetector


class InputPipeline:
    """Consumes incoming client audio frames, detects turn boundaries via
    TurnDetector, and hands off each closed turn's audio to a
    transcription callback. Also exposes the current in-progress speech
    run length, for barge-in detection while the assistant is speaking."""

    def __init__(
        self,
        transcribe_fn: Callable[[np.ndarray], tuple[str, str]],
        on_turn_transcribed: Callable[[str, str, int, int], None],
    ):
        self._transcribe = transcribe_fn
        self._on_turn = on_turn_transcribed
        self._detector = TurnDetector()
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=TurnDetector.SAMPLE_RATE)
        self._turn_start_ms: int | None = None
        self._elapsed_ms = 0

    def handle_frame(self, frame: av.AudioFrame) -> None:
        for resampled in self._resampler.resample(frame):
            pcm = resampled.to_ndarray().astype(np.int16).tobytes()
            frame_ms = int(resampled.samples / resampled.sample_rate * 1000)
            if self._turn_start_ms is None:
                self._turn_start_ms = self._elapsed_ms
            closed_turn = self._detector.feed(pcm)
            self._elapsed_ms += frame_ms
            if closed_turn is not None:
                start_ms = self._turn_start_ms
                end_ms = self._elapsed_ms
                self._turn_start_ms = None
                audio_np = np.frombuffer(closed_turn, dtype=np.int16).astype(np.float32) / 32768.0
                text, language = self._transcribe(audio_np)
                text = text.strip()
                if text:
                    self._on_turn(text, language, start_ms, end_ms)

    def current_speech_run_ms(self) -> int:
        return self._detector.speech_run_ms
```

(This is the file's existing content — from the core pipeline plan, which
already added the `(text, language)` tuple return from `transcribe_fn` —
with only the final `current_speech_run_ms` method added. If the file you
read differs from this, the codebase has moved since this plan was
written; match your edits to what's actually there rather than
overwriting unrelated changes.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_vad.py tests/serverlocal/test_input_pipeline.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 6: Commit**

```bash
git add serverlocal/vad.py serverlocal/input_pipeline.py tests/serverlocal/test_vad.py tests/serverlocal/test_input_pipeline.py
git commit -m "Track consecutive speech duration for barge-in detection"
```

---

### Task 2: Delegation marker detection + event builder

**Files:**
- Modify: `serverlocal/events.py`
- Modify: `tests/serverlocal/test_events.py`
- Create: `serverlocal/delegation.py`
- Create: `tests/serverlocal/test_delegation.py`

**Interfaces:**
- Produces: `events.delegation_created(delegation_id: str) -> dict`,
  `delegation.strip_delegation_markers(token_stream: Iterable[str],
  on_delegation: Callable[[str], None]) -> Iterator[str]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/serverlocal/test_events.py`:

```python
def test_delegation_created_shape():
    event = events.delegation_created("abc-123")
    assert event == {
        "type": "session.delegation.created",
        "delegation": {"target": "client", "id": "abc-123"},
    }
```

Create `tests/serverlocal/test_delegation.py`:

```python
from serverlocal.delegation import strip_delegation_markers


def test_passes_through_text_with_no_marker():
    calls = []
    result = list(strip_delegation_markers(["Hello", " world."], calls.append))
    assert "".join(result) == "Hello world."
    assert calls == []


def test_extracts_a_marker_contained_in_one_token():
    calls = []
    result = list(strip_delegation_markers(["[[SEARCH: capital of Peru]]"], calls.append))
    assert "".join(result) == ""
    assert calls == ["capital of Peru"]


def test_extracts_a_marker_split_across_many_tokens():
    calls = []
    tokens = ["Sure, ", "[[SEARCH", ": weather in ", "Lima today", "]]", " done."]
    result = list(strip_delegation_markers(tokens, calls.append))
    assert "".join(result) == "Sure,  done."
    assert calls == ["weather in Lima today"]


def test_withholds_a_possible_partial_marker_until_the_stream_ends():
    calls = []
    # "[[SEA" alone could still become a marker -- must not be flushed
    # as plain text until the stream proves it isn't one.
    gen = strip_delegation_markers(iter(["before ", "[[SEA"]), calls.append)
    first = next(gen)
    assert first == "before "
    remainder = list(gen)
    # The stream ends without ever completing "]]" -- the partial text
    # is flushed as plain text rather than silently dropped.
    assert "".join(remainder) == "[[SEA"
    assert calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_events.py tests/serverlocal/test_delegation.py -v`
Expected: FAIL — `delegation_created` doesn't exist;
`ModuleNotFoundError: No module named 'serverlocal.delegation'`.

- [ ] **Step 3: Add `delegation_created` to `serverlocal/events.py`**

Add this function to the existing file (alongside `session_created`,
`error`, etc. — do not otherwise modify the file):

```python
def delegation_created(delegation_id: str) -> dict:
    return {"type": "session.delegation.created", "delegation": {"target": "client", "id": delegation_id}}
```

- [ ] **Step 4: Write `serverlocal/delegation.py`**

```python
import re
from collections.abc import Callable, Iterable, Iterator

SEARCH_MARKER = re.compile(r"\[\[SEARCH:\s*(.*?)\]\]")


def strip_delegation_markers(token_stream: Iterable[str], on_delegation: Callable[[str], None]) -> Iterator[str]:
    """Wraps a streamed-token generator, detecting [[SEARCH: query]]
    markers that may span multiple tokens. Calls on_delegation(query) for
    each complete marker found and yields the token stream with markers
    removed. Text that looks like it MIGHT be the start of a marker is
    withheld until either the marker completes (and is stripped) or the
    stream proves it wasn't one (flushed as plain text, including at
    stream end)."""
    buffer = ""
    for token in token_stream:
        buffer += token
        while True:
            match = SEARCH_MARKER.search(buffer)
            if not match:
                break
            on_delegation(match.group(1).strip())
            buffer = buffer[: match.start()] + buffer[match.end() :]
        safe_end = _safe_flush_point(buffer)
        if safe_end:
            yield buffer[:safe_end]
            buffer = buffer[safe_end:]
    if buffer:
        yield buffer


def _safe_flush_point(buffer: str) -> int:
    """How much of buffer can be safely yielded without risking splitting
    a still-forming marker. Withholds everything from the last "[[" if it
    hasn't yet resolved into a confirmed non-marker or a completed one."""
    idx = buffer.rfind("[[")
    if idx == -1:
        return len(buffer)
    return idx
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_events.py tests/serverlocal/test_delegation.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 6: Commit**

```bash
git add serverlocal/events.py serverlocal/delegation.py tests/serverlocal/test_events.py tests/serverlocal/test_delegation.py
git commit -m "Add delegation marker detection and session.delegation.created builder"
```

---

### Task 3: Web search tool

**Files:**
- Create: `serverlocal/search_tool.py`
- Create: `tests/serverlocal/test_search_tool.py`
- Modify: `serverlocal/requirements.txt`

**Interfaces:**
- Produces: `search_tool.web_search(query: str, max_results: int = 3) ->
  list[dict]` (each dict: `{"url": str, "title": str, "body": str}`),
  `search_tool.WEB_SEARCH_TOOL: dict` (Ollama/OpenAI-style function-tool
  definition).

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_search_tool.py
from unittest.mock import MagicMock, patch

from serverlocal import search_tool


def test_web_search_returns_url_title_and_body(monkeypatch):
    fake_results = [
        {"href": "https://example.com/a", "title": "A", "body": "About A"},
        {"href": "https://example.com/b", "title": "B", "body": "About B"},
    ]
    fake_ddgs = MagicMock()
    fake_ddgs.__enter__.return_value.text.return_value = iter(fake_results)
    monkeypatch.setattr(search_tool, "DDGS", lambda: fake_ddgs)

    results = search_tool.web_search("test query", max_results=2)

    assert results == [
        {"url": "https://example.com/a", "title": "A", "body": "About A"},
        {"url": "https://example.com/b", "title": "B", "body": "About B"},
    ]
    fake_ddgs.__enter__.return_value.text.assert_called_once_with("test query", max_results=2)


def test_web_search_drops_results_with_no_url(monkeypatch):
    fake_results = [{"href": "", "title": "No URL", "body": "x"}]
    fake_ddgs = MagicMock()
    fake_ddgs.__enter__.return_value.text.return_value = iter(fake_results)
    monkeypatch.setattr(search_tool, "DDGS", lambda: fake_ddgs)

    assert search_tool.web_search("q") == []


def test_web_search_returns_empty_list_on_failure(monkeypatch):
    def raise_error():
        raise RuntimeError("network down")

    monkeypatch.setattr(search_tool, "DDGS", raise_error)

    assert search_tool.web_search("q") == []


def test_web_search_tool_definition_matches_ollama_function_tool_shape():
    assert search_tool.WEB_SEARCH_TOOL["type"] == "function"
    function = search_tool.WEB_SEARCH_TOOL["function"]
    assert function["name"] == "web_search"
    assert function["parameters"]["required"] == ["query"]
    assert "query" in function["parameters"]["properties"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_search_tool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'serverlocal.search_tool'`.

- [ ] **Step 3: Write `serverlocal/search_tool.py`**

```python
from duckduckgo_search import DDGS

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information needed to answer the user's question.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Runs a DuckDuckGo text search. Returns an empty list on any
    failure (network down, DuckDuckGo rate limit, etc) rather than
    raising -- a failed search should degrade the model's answer, not
    crash the request."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []
    return [
        {"url": r["href"], "title": r.get("title", ""), "body": r.get("body", "")}
        for r in results
        if r.get("href")
    ]
```

- [ ] **Step 4: Update `serverlocal/requirements.txt`**

Add one line (keep the existing comment block and every other line
unchanged):

```
duckduckgo-search
```

- [ ] **Step 5: Install and run tests to verify they pass**

Run: `.venv/bin/pip install duckduckgo-search`
Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_search_tool.py -v`
Expected: PASS (4/4)

- [ ] **Step 6: Commit**

```bash
git add serverlocal/search_tool.py tests/serverlocal/test_search_tool.py serverlocal/requirements.txt
git commit -m "Add DuckDuckGo web_search tool"
```

---

### Task 4: Ollama chat/tool-calling/schema wrapper

**Files:**
- Create: `serverlocal/ollama_responses.py`
- Create: `tests/serverlocal/test_ollama_responses.py`

**Interfaces:**
- Consumes: `search_tool.WEB_SEARCH_TOOL`, `search_tool.web_search` (Task 3).
- Produces: `ollama_responses.ResponseResult` (dataclass: `text: str`,
  `sources: list[dict]`, `input_tokens: int`, `output_tokens: int`,
  `searches: int`), `ollama_responses.generate(instructions: str,
  user_text: str, schema: dict | None, want_search: bool, model: str =
  "qwen2.5:7b") -> ResponseResult`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_ollama_responses.py
from unittest.mock import MagicMock

from serverlocal import ollama_responses


def _fake_response(content, prompt_eval_count=10, eval_count=5, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message, "prompt_eval_count": prompt_eval_count, "eval_count": eval_count}


def test_generate_without_search_makes_one_call_and_returns_its_content(monkeypatch):
    fake_post = MagicMock()
    fake_post.return_value.json.return_value = _fake_response("Hola mundo")
    fake_post.return_value.raise_for_status.return_value = None
    monkeypatch.setattr(ollama_responses.requests, "post", fake_post)

    result = ollama_responses.generate("Be brief.", "hola", schema=None, want_search=False, model="qwen2.5:7b")

    assert result.text == "Hola mundo"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.searches == 0
    assert result.sources == []
    assert fake_post.call_count == 1
    body = fake_post.call_args.kwargs["json"]
    assert body["model"] == "qwen2.5:7b"
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hola"}]
    assert "tools" not in body


def test_generate_with_schema_passes_it_as_format(monkeypatch):
    fake_post = MagicMock()
    fake_post.return_value.json.return_value = _fake_response('{"answer": "42"}')
    fake_post.return_value.raise_for_status.return_value = None
    monkeypatch.setattr(ollama_responses.requests, "post", fake_post)

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = ollama_responses.generate("", "question", schema=schema, want_search=False, model="qwen2.5:7b")

    assert result.text == '{"answer": "42"}'
    body = fake_post.call_args.kwargs["json"]
    assert body["format"] == schema


def test_generate_with_search_calls_tool_once_then_returns_final_answer(monkeypatch):
    tool_call = {"function": {"name": "web_search", "arguments": {"query": "capital of Peru"}}}
    first = _fake_response("", tool_calls=[tool_call])
    final = _fake_response("The capital of Peru is Lima.", prompt_eval_count=20, eval_count=8)
    fake_post = MagicMock()
    fake_post.return_value.raise_for_status.return_value = None
    fake_post.return_value.json.side_effect = [first, final]
    monkeypatch.setattr(ollama_responses.requests, "post", fake_post)

    fake_search = MagicMock(return_value=[{"url": "https://example.com", "title": "Peru", "body": "Lima is..."}])
    monkeypatch.setattr(ollama_responses.search_tool, "web_search", fake_search)

    result = ollama_responses.generate("", "what is the capital of Peru?", schema=None, want_search=True, model="qwen2.5:7b")

    assert result.text == "The capital of Peru is Lima."
    assert result.searches == 1
    assert result.sources == [{"url": "https://example.com", "title": "Peru", "body": "Lima is..."}]
    assert result.input_tokens == 20
    assert result.output_tokens == 8
    fake_search.assert_called_once_with("capital of Peru")
    assert fake_post.call_count == 2
    first_body = fake_post.call_args_list[0].kwargs["json"]
    assert first_body["tools"] == [ollama_responses.search_tool.WEB_SEARCH_TOOL]
    second_body = fake_post.call_args_list[1].kwargs["json"]
    assert "tools" not in second_body
    assert second_body["messages"][-1]["role"] == "tool"


def test_generate_with_search_but_model_declines_to_call_it(monkeypatch):
    fake_post = MagicMock()
    fake_post.return_value.raise_for_status.return_value = None
    fake_post.return_value.json.return_value = _fake_response("I don't need to search for that.")
    monkeypatch.setattr(ollama_responses.requests, "post", fake_post)

    result = ollama_responses.generate("", "hello", schema=None, want_search=True, model="qwen2.5:7b")

    assert result.text == "I don't need to search for that."
    assert result.searches == 0
    assert fake_post.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_ollama_responses.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'serverlocal.ollama_responses'`.

- [ ] **Step 3: Write `serverlocal/ollama_responses.py`**

```python
from dataclasses import dataclass, field

import requests

from . import search_tool

OLLAMA_HOST = "http://localhost:11434"


@dataclass
class ResponseResult:
    text: str
    sources: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0


def _chat(messages: list[dict], model: str, format_schema: dict | None = None, tools: list[dict] | None = None) -> dict:
    payload: dict = {"model": model, "messages": messages, "stream": False}
    if format_schema is not None:
        payload["format"] = format_schema
    if tools is not None:
        payload["tools"] = tools
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _format_search_results(results: list[dict]) -> str:
    if not results:
        return "No search results found."
    lines = [f"- {r['title']}: {r['body']} ({r['url']})" for r in results]
    return "Search results:\n" + "\n".join(lines)


def generate(instructions: str, user_text: str, schema: dict | None, want_search: bool, model: str = "qwen2.5:7b") -> ResponseResult:
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": user_text})

    sources: list[dict] = []
    searches = 0

    if want_search:
        first = _chat(messages, model, tools=[search_tool.WEB_SEARCH_TOOL])
        tool_calls = first.get("message", {}).get("tool_calls") or []
        if tool_calls:
            query = tool_calls[0].get("function", {}).get("arguments", {}).get("query", user_text)
            results = search_tool.web_search(query)
            searches = 1
            sources = results
            messages.append(first["message"])
            messages.append({"role": "tool", "content": _format_search_results(results)})
            final = _chat(messages, model, format_schema=schema)
        elif schema is not None:
            final = _chat(messages, model, format_schema=schema)
        else:
            final = first
    else:
        final = _chat(messages, model, format_schema=schema)

    return ResponseResult(
        text=final.get("message", {}).get("content", ""),
        sources=sources,
        input_tokens=final.get("prompt_eval_count", 0),
        output_tokens=final.get("eval_count", 0),
        searches=searches,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_ollama_responses.py -v`
Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add serverlocal/ollama_responses.py tests/serverlocal/test_ollama_responses.py
git commit -m "Add Ollama chat wrapper with tool-calling and schema support"
```

---

### Task 5: `/responses` route + server wiring

**Files:**
- Create: `serverlocal/responses_api.py`
- Create: `tests/serverlocal/test_responses_api.py`
- Modify: `serverlocal/server.py`

**Interfaces:**
- Consumes: `ollama_responses.generate`, `ollama_responses.ResponseResult` (Task 4).
- Produces: FastAPI `router` with `POST /responses`, matching
  `decodeTeachingResponse`'s exact expected JSON shape (see Global
  Constraints).

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_responses_api.py
from unittest.mock import MagicMock

from serverlocal import responses_api
from serverlocal.ollama_responses import ResponseResult


def test_responses_route_builds_decodeteachingresponse_compatible_shape(monkeypatch):
    fake_generate = MagicMock(return_value=ResponseResult(text="Hola", input_tokens=12, output_tokens=7))
    monkeypatch.setattr(responses_api, "generate", fake_generate)

    body = responses_api.ResponsesBody(
        model="qwen2.5:7b",
        instructions="Reply briefly.",
        input=[{"role": "user", "content": "hola"}],
    )

    import asyncio

    result = asyncio.run(responses_api.responses_route(body))

    assert result["status"] == "completed"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 7}
    content = result["output"][-1]["content"][0]
    assert content["type"] == "output_text"
    assert content["text"] == "Hola"
    assert content["annotations"] == []
    assert not any(item.get("type") == "web_search_call" for item in result["output"])
    fake_generate.assert_called_once_with("Reply briefly.", "hola", None, False, "qwen2.5:7b")


def test_responses_route_extracts_schema_and_search_flag(monkeypatch):
    fake_generate = MagicMock(return_value=ResponseResult(text="ok"))
    monkeypatch.setattr(responses_api, "generate", fake_generate)

    schema = {"type": "object", "properties": {}}
    body = responses_api.ResponsesBody(
        model="qwen2.5:7b",
        instructions="",
        input=[{"role": "user", "content": "q"}],
        text={"format": {"type": "json_schema", "name": "mural_result", "strict": True, "schema": schema}},
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        max_tool_calls=1,
    )

    import asyncio

    asyncio.run(responses_api.responses_route(body))

    fake_generate.assert_called_once_with("", "q", schema, True, "qwen2.5:7b")


def test_responses_route_includes_web_search_call_marker_and_citations_when_searched(monkeypatch):
    fake_generate = MagicMock(return_value=ResponseResult(
        text="Lima is the capital.",
        sources=[{"url": "https://example.com/peru", "title": "Peru"}],
        searches=1,
    ))
    monkeypatch.setattr(responses_api, "generate", fake_generate)

    body = responses_api.ResponsesBody(model="qwen2.5:7b", instructions="", input=[{"role": "user", "content": "q"}])

    import asyncio

    result = asyncio.run(responses_api.responses_route(body))

    assert result["output"][0] == {"type": "web_search_call"}
    annotations = result["output"][1]["content"][0]["annotations"]
    assert annotations == [{"type": "url_citation", "url": "https://example.com/peru", "title": "Peru"}]


def test_responses_route_joins_multiple_string_input_items(monkeypatch):
    fake_generate = MagicMock(return_value=ResponseResult(text="ok"))
    monkeypatch.setattr(responses_api, "generate", fake_generate)

    body = responses_api.ResponsesBody(
        model="qwen2.5:7b",
        instructions="",
        input=[{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
    )

    import asyncio

    asyncio.run(responses_api.responses_route(body))

    fake_generate.assert_called_once_with("", "first\nsecond", None, False, "qwen2.5:7b")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_responses_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'serverlocal.responses_api'`.

- [ ] **Step 3: Write `serverlocal/responses_api.py`**

```python
from fastapi import APIRouter
from pydantic import BaseModel

from .ollama_responses import generate

router = APIRouter()


class ResponsesBody(BaseModel):
    model: str = "qwen2.5:7b"
    instructions: str = ""
    input: list[dict]
    store: bool | None = None
    max_output_tokens: int | None = None
    reasoning: dict | None = None
    text: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | None = None
    max_tool_calls: int | None = None


def _extract_user_text(input_items: list[dict]) -> str:
    return "\n".join(item["content"] for item in input_items if isinstance(item.get("content"), str))


def _extract_schema(body: ResponsesBody) -> dict | None:
    if not body.text:
        return None
    fmt = body.text.get("format") or {}
    if fmt.get("type") != "json_schema":
        return None
    return fmt.get("schema")


def _wants_search(body: ResponsesBody) -> bool:
    return bool(body.tools) and any(t.get("type") == "web_search" for t in body.tools)


@router.post("/responses")
async def responses_route(body: ResponsesBody) -> dict:
    user_text = _extract_user_text(body.input)
    schema = _extract_schema(body)
    want_search = _wants_search(body)

    result = generate(body.instructions, user_text, schema, want_search, body.model)

    output: list[dict] = []
    if result.searches:
        output.append({"type": "web_search_call"})
    output.append({
        "content": [{
            "type": "output_text",
            "text": result.text,
            "annotations": [
                {"type": "url_citation", "url": source["url"], "title": source.get("title", "")}
                for source in result.sources
            ],
        }],
    })

    return {
        "status": "completed",
        "output": output,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
    }
```

- [ ] **Step 4: Wire the router into `serverlocal/server.py`**

The file currently reads:

```python
from . import models
from .session import router

app = FastAPI()
app.include_router(router)
```

Change it to include both routers:

```python
from . import models
from .responses_api import router as responses_router
from .session import router as session_router

app = FastAPI()
app.include_router(session_router)
app.include_router(responses_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_responses_api.py -v`
Expected: PASS (4/4)

Run: `.venv/bin/python3 -c "import serverlocal.server"` — expected: no
traceback (both routers import cleanly; model loading is still deferred
to the startup hook).

- [ ] **Step 6: Commit**

```bash
git add serverlocal/responses_api.py tests/serverlocal/test_responses_api.py serverlocal/server.py
git commit -m "Add POST /responses endpoint and wire it into the FastAPI app"
```

---

### Task 6: Barge-in + delegation wiring in `session.py`

**Files:**
- Modify: `serverlocal/session.py`
- Modify: `tests/serverlocal/test_session.py`

**Interfaces:**
- Consumes: `InputPipeline.current_speech_run_ms()` (Task 1),
  `delegation.strip_delegation_markers`, `events.delegation_created`
  (Task 2).
- Produces: `Session.response_task: asyncio.Task | None`,
  `Session.pending_delegations: set[str]`,
  `_cancel_current_response(session) -> None`,
  `speak_commentary(session, text) -> Awaitable[None]` — Task 7's message
  handler calls both of these.

Read the current `serverlocal/session.py` in full before editing — this
task and Task 7 both modify it; work from what's actually on disk, not
just what's quoted here (Tasks 1-5 didn't touch this file, so it should
match what's shown, but confirm).

- [ ] **Step 1: Write the failing tests**

Add to `tests/serverlocal/test_session.py`:

```python
def test_cancel_current_response_cancels_task_and_clears_track():
    session = Session(pc=None)

    class FakeTrack:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    session.output_track = FakeTrack()

    async def scenario():
        async def never_finishes():
            await asyncio.sleep(3600)

        session.response_task = asyncio.ensure_future(never_finishes())
        await asyncio.sleep(0)  # let the task actually start
        session_module._cancel_current_response(session)
        await asyncio.sleep(0)
        return session.response_task.cancelled(), session.output_track.cleared

    cancelled, cleared = asyncio.run(scenario())
    assert cancelled is True
    assert cleared is True


def test_cancel_current_response_is_a_no_op_when_nothing_is_running():
    session = Session(pc=None)

    class FakeTrack:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    session.output_track = FakeTrack()
    session_module._cancel_current_response(session)  # must not raise
    assert session.output_track.cleared is True


def test_consume_audio_cancels_response_task_on_sustained_barge_in_speech(monkeypatch):
    session = Session(pc=None)
    sent = []
    session.send = sent.append

    class FakeTrack:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    session.output_track = FakeTrack()

    async def never_finishes():
        await asyncio.sleep(3600)

    async def scenario():
        session.response_task = asyncio.ensure_future(never_finishes())
        await asyncio.sleep(0)

        class FakeInputPipeline:
            def __init__(self):
                self._calls = 0

            def handle_frame(self, frame):
                self._calls += 1

            def current_speech_run_ms(self):
                # First frame: below threshold. Second: at/above 200ms.
                return 100 if self._calls == 1 else 220

        class FakeTrackSource:
            def __init__(self):
                self._frames = [object(), object()]

            async def recv(self):
                if not self._frames:
                    raise RuntimeError("no more frames")
                return self._frames.pop(0)

        await session_module._consume_audio(FakeTrackSource(), FakeInputPipeline(), session)
        return session.response_task.cancelled(), session.output_track.cleared

    cancelled, cleared = asyncio.run(scenario())
    assert cancelled is True
    assert cleared is True


def test_on_turn_transcribed_cancels_any_still_running_response_before_starting_a_new_one(monkeypatch):
    monkeypatch.setattr(session_module.asyncio, "ensure_future", lambda coro: coro.close())
    session = Session(pc=None)
    sent = []
    session.send = sent.append
    cancelled = {"value": False}

    class FakeTask:
        def done(self):
            return False

        def cancel(self):
            cancelled["value"] = True

    class FakeTrack:
        def clear(self):
            pass

    session.response_task = FakeTask()
    session.output_track = FakeTrack()

    pipeline = build_input_pipeline(session)
    pipeline._on_turn("hola de nuevo", "es", 0, 500)

    assert cancelled["value"] is True


def test_run_response_turn_strips_delegation_marker_and_emits_delegation_created(monkeypatch):
    monkeypatch.setattr(ollama_client, "stream_reply", lambda prompt: iter(["[[SEARCH: capital of Peru]]"]))
    sent = []
    session = Session(pc=None)
    session.send = sent.append

    class FakeTrack:
        async def push_pcm(self, samples, sample_rate):
            pass

    session.output_track = FakeTrack()

    asyncio.run(run_response_turn(session, "what's the capital of Peru?"))

    delegation_events = [e for e in sent if e["type"] == "session.delegation.created"]
    assert len(delegation_events) == 1
    delegation_id = delegation_events[0]["delegation"]["id"]
    assert delegation_id in session.pending_delegations
    # The marker itself must never be sent as spoken output.
    assert not any(e["type"] == "session.output_transcript.delta" for e in sent)


def test_speak_commentary_synthesizes_and_sends_output_delta():
    monkeypatch_models = models.synthesize
    try:
        import numpy as np

        models.synthesize = lambda text, voice, lang: (np.zeros(1600, dtype=np.int16), 16000)
        session = Session(pc=None)
        sent = []
        session.send = sent.append

        class FakeTrack:
            async def push_pcm(self, samples, sample_rate):
                pass

        session.output_track = FakeTrack()

        asyncio.run(session_module.speak_commentary(session, "Lima is the capital of Peru."))

        deltas = [e for e in sent if e["type"] == "session.output_transcript.delta"]
        assert len(deltas) == 1
        assert deltas[0]["delta"] == "Lima is the capital of Peru."
    finally:
        models.synthesize = monkeypatch_models
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_session.py -v -k "cancel_current_response or barge_in or on_turn_transcribed_cancels or delegation_marker or speak_commentary"`
Expected: FAIL — `AttributeError`/`ImportError` on the not-yet-added
names (`_cancel_current_response`, `speak_commentary`,
`session.response_task`, `session.pending_delegations`).

- [ ] **Step 3: Update `serverlocal/session.py`**

Add these imports at the top (alongside the existing ones):

```python
from .delegation import strip_delegation_markers
```

Add a constant near `DEFAULT_SYSTEM_PROMPT`:

```python
BARGE_IN_THRESHOLD_MS = 200
SEARCH_MARKER_INSTRUCTION = (
    "\nIf you need current information you don't already know, reply with "
    "exactly [[SEARCH: <query>]] and nothing else -- no other text."
)
```

Update `Session.__init__` to add two new attributes (keep every existing
line unchanged, just add these):

```python
    def __init__(self, pc: RTCPeerConnection | None):
        self.id = str(uuid.uuid4())
        self.pc = pc
        self.channel = None
        self.output_track = OutputAudioTrack()
        self.started_at = time.monotonic()
        self.instructions: str = ""
        self.language: str | None = None
        self.response_task: asyncio.Task | None = None
        self.pending_delegations: set[str] = set()
```

Add a module-level helper function (near the top, after `Session`):

```python
def _cancel_current_response(session: Session) -> None:
    if session.response_task is not None and not session.response_task.done():
        session.response_task.cancel()
    session.output_track.clear()
```

Update `build_input_pipeline`'s `on_turn_transcribed` closure to cancel
any still-running response before starting a new one, and to store the
new task's handle:

```python
    def on_turn_transcribed(text: str, language: str, start_ms: int, end_ms: int) -> None:
        session.language = language
        session.send(events.transcript_delta("input", text, start_ms, end_ms))
        _cancel_current_response(session)
        session.response_task = asyncio.ensure_future(run_response_turn(session, text))
```

Update `run_response_turn` to append the search-marker instruction and
wrap `stream_reply_fn` with delegation-marker stripping:

```python
async def run_response_turn(session: Session, user_text: str) -> None:
    system_prompt = session.instructions if session.instructions else DEFAULT_SYSTEM_PROMPT
    prompt = f"{system_prompt}{SEARCH_MARKER_INSTRUCTION}\nUser: {user_text}\nAssistant:"

    def stream_reply_fn(p: str):
        def on_delegation(query: str) -> None:
            delegation_id = str(uuid.uuid4())
            session.pending_delegations.add(delegation_id)
            session.send(events.delegation_created(delegation_id))

        return strip_delegation_markers(ollama_client.stream_reply(p), on_delegation)

    def synthesize_fn(text: str):
        voice, lang = _voice_for_language(session.language)
        return models.synthesize(text, voice=voice, lang=lang)

    async def push_audio_fn(samples, sample_rate):
        await session.output_track.push_pcm(samples, sample_rate)

    def on_output_delta(text: str, start_ms: int, end_ms: int) -> None:
        session.send(events.transcript_delta("output", text, start_ms, end_ms))

    pipeline = ResponsePipeline(stream_reply_fn, synthesize_fn, push_audio_fn, on_output_delta)
    try:
        await pipeline.run(prompt)
    except Exception as exc:
        session.send(events.error(str(exc)))
```

(Note: `chunk_sentences` inside `ResponsePipeline` only yields *complete*
sentences, and `strip_delegation_markers` runs before that — a
marker-only reply like `"[[SEARCH: capital of Peru]]"` produces zero
non-empty text for `chunk_sentences` to chunk, so no
`output_transcript.delta`/audio is produced for it, matching the test's
expectation.)

Add a new function `speak_commentary`, near `run_response_turn`:

```python
async def speak_commentary(session: Session, text: str) -> None:
    def stream_reply_fn(_p: str):
        return iter([text])

    def synthesize_fn(t: str):
        voice, lang = _voice_for_language(session.language)
        return models.synthesize(t, voice=voice, lang=lang)

    async def push_audio_fn(samples, sample_rate):
        await session.output_track.push_pcm(samples, sample_rate)

    def on_output_delta(t: str, start_ms: int, end_ms: int) -> None:
        session.send(events.transcript_delta("output", t, start_ms, end_ms))

    pipeline = ResponsePipeline(stream_reply_fn, synthesize_fn, push_audio_fn, on_output_delta)
    try:
        await pipeline.run(text)
    except Exception as exc:
        session.send(events.error(str(exc)))
```

Update `_consume_audio` to check for barge-in after each frame:

```python
async def _consume_audio(track, input_pipeline: InputPipeline, session: Session) -> None:
    while True:
        try:
            frame = await track.recv()
        except Exception:
            break
        try:
            input_pipeline.handle_frame(frame)
        except Exception as exc:
            session.send(events.error(str(exc)))
            continue
        if (
            session.response_task is not None
            and not session.response_task.done()
            and input_pipeline.current_speech_run_ms() >= BARGE_IN_THRESHOLD_MS
        ):
            _cancel_current_response(session)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_session.py -v`
Expected: PASS (all tests in the file, including every test from the
core pipeline plan plus the new ones added in this task)

- [ ] **Step 5: Commit**

```bash
git add serverlocal/session.py tests/serverlocal/test_session.py
git commit -m "Add barge-in cancellation and delegation marker/commentary wiring"
```

---

### Task 7: Data-channel message handler + session lifecycle

**Files:**
- Modify: `serverlocal/session.py`
- Modify: `tests/serverlocal/test_session.py`

**Interfaces:**
- Produces: `Session.pending_context: list[str]`, `Session.muted: bool`,
  `Session.usage_task: asyncio.Task | None`,
  `_handle_client_message(session, message) -> None`,
  `_teardown_session(session, send_closed_event: bool) ->
  Awaitable[None]`, `_broadcast_usage(session) -> Awaitable[None]`.

Read the current `serverlocal/session.py` in full before editing (it now
includes Task 6's changes).

- [ ] **Step 1: Write the failing tests**

Add to `tests/serverlocal/test_session.py`:

```python
def test_handle_instructions_and_thinking_append_queue_pending_context():
    session = Session(pc=None)
    session_module._handle_client_message(session, json.dumps({
        "type": "session.instructions.append", "content": "Discuss only fruit."
    }))
    session_module._handle_client_message(session, json.dumps({
        "type": "session.thinking.append", "content": "Learner is at level 2."
    }))
    assert session.pending_context == ["Discuss only fruit.", "Learner is at level 2."]


def test_handle_mute_and_unmute_toggle_session_muted():
    session = Session(pc=None)
    assert session.muted is False
    session_module._handle_client_message(session, json.dumps({"type": "session.input_audio.mute"}))
    assert session.muted is True
    session_module._handle_client_message(session, json.dumps({"type": "session.input_audio.unmute"}))
    assert session.muted is False


def test_handle_commentary_append_cancels_current_response_and_speaks_it(monkeypatch):
    session = Session(pc=None)
    session.pending_delegations.add("deleg-1")
    scheduled = {}
    monkeypatch.setattr(session_module.asyncio, "ensure_future", lambda coro: scheduled.setdefault("coro", coro) or coro.close())

    session_module._handle_client_message(session, json.dumps({
        "type": "session.commentary.append", "content": "Here's the answer.", "delegation_id": "deleg-1"
    }))

    assert "deleg-1" not in session.pending_delegations
    assert "coro" in scheduled


def test_handle_unknown_message_type_is_ignored_without_raising():
    session = Session(pc=None)
    session_module._handle_client_message(session, json.dumps({"type": "session.something.unrecognized"}))
    session_module._handle_client_message(session, "not even json")
    session_module._handle_client_message(session, 12345)  # not a string at all


def test_teardown_session_sends_closed_event_cancels_tasks_and_removes_from_registry():
    session = Session(pc=None)
    sent = []
    session.send = sent.append
    session_module._active_sessions[session.id] = session

    class FakeTrack:
        def clear(self):
            pass

    session.output_track = FakeTrack()

    async def never_finishes():
        await asyncio.sleep(3600)

    async def scenario():
        session.response_task = asyncio.ensure_future(never_finishes())
        session.usage_task = asyncio.ensure_future(never_finishes())
        await asyncio.sleep(0)
        await session_module._teardown_session(session, send_closed_event=True)
        return session.response_task.cancelled(), session.usage_task.cancelled()

    response_cancelled, usage_cancelled = asyncio.run(scenario())
    assert response_cancelled is True
    assert usage_cancelled is True
    assert session.id not in session_module._active_sessions
    closed_events = [e for e in sent if e["type"] == "session.closed"]
    assert len(closed_events) == 1
    assert closed_events[0]["usage"]["seconds"] >= 0


def test_teardown_session_without_closed_event_sends_nothing():
    session = Session(pc=None)
    sent = []
    session.send = sent.append

    class FakeTrack:
        def clear(self):
            pass

    session.output_track = FakeTrack()

    asyncio.run(session_module._teardown_session(session, send_closed_event=False))
    assert sent == []


def test_broadcast_usage_sends_periodic_updates_until_cancelled(monkeypatch):
    session = Session(pc=None)
    sent = []
    session.send = sent.append
    monkeypatch.setattr(session_module, "USAGE_BROADCAST_INTERVAL_SECONDS", 0.01)

    async def scenario():
        task = asyncio.ensure_future(session_module._broadcast_usage(session))
        await asyncio.sleep(0.035)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    usage_events = [e for e in sent if e["type"] == "session.usage.updated"]
    assert len(usage_events) >= 2
```

Add `import json` to the test file's imports if it isn't already there
(check the existing imports first — `test_session.py` may already import
it for building event payloads).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_session.py -v -k "handle_instructions or handle_mute or handle_commentary or handle_unknown or teardown_session or broadcast_usage"`
Expected: FAIL — `AttributeError`/`ImportError` on the not-yet-added
names.

- [ ] **Step 3: Update `serverlocal/session.py`**

Add `import json` if not already present at the top (it already is, from
the core pipeline plan's `Session.send`).

Add a constant near `BARGE_IN_THRESHOLD_MS`:

```python
USAGE_BROADCAST_INTERVAL_SECONDS = 5
```

Add a module-level registry near `router = APIRouter()`:

```python
_active_sessions: dict[str, "Session"] = {}
```

Update `Session.__init__` to add three more attributes:

```python
        self.response_task: asyncio.Task | None = None
        self.pending_delegations: set[str] = set()
        self.pending_context: list[str] = []
        self.muted: bool = False
        self.usage_task: asyncio.Task | None = None
```

(This replaces the two-line addition from Task 6 with these five lines —
`response_task`/`pending_delegations` are the same two attributes Task 6
added, now joined by the three this task adds.)

Update `run_response_turn` to consume `pending_context` when building the
prompt (this changes the `prompt = ...` line Task 6 added):

```python
async def run_response_turn(session: Session, user_text: str) -> None:
    system_prompt = session.instructions if session.instructions else DEFAULT_SYSTEM_PROMPT
    context_note = ""
    if session.pending_context:
        context_note = "\n" + "\n".join(session.pending_context)
        session.pending_context = []
    prompt = f"{system_prompt}{SEARCH_MARKER_INSTRUCTION}{context_note}\nUser: {user_text}\nAssistant:"

    def stream_reply_fn(p: str):
        def on_delegation(query: str) -> None:
            delegation_id = str(uuid.uuid4())
            session.pending_delegations.add(delegation_id)
            session.send(events.delegation_created(delegation_id))

        return strip_delegation_markers(ollama_client.stream_reply(p), on_delegation)

    def synthesize_fn(text: str):
        voice, lang = _voice_for_language(session.language)
        return models.synthesize(text, voice=voice, lang=lang)

    async def push_audio_fn(samples, sample_rate):
        await session.output_track.push_pcm(samples, sample_rate)

    def on_output_delta(text: str, start_ms: int, end_ms: int) -> None:
        session.send(events.transcript_delta("output", text, start_ms, end_ms))

    pipeline = ResponsePipeline(stream_reply_fn, synthesize_fn, push_audio_fn, on_output_delta)
    try:
        await pipeline.run(prompt)
    except Exception as exc:
        session.send(events.error(str(exc)))
```

(Everything from `def stream_reply_fn` onward is identical to what Task 6
wrote — only the `prompt = ...` line changed, to fold in
`context_note`. Shown in full here so this task's implementer doesn't
need Task 6's brief open side by side.)

Add the message handler and lifecycle functions (near `_consume_audio`):

```python
def _handle_client_message(session: Session, message) -> None:
    if not isinstance(message, str):
        return
    try:
        event = json.loads(message)
    except (ValueError, TypeError):
        return
    event_type = event.get("type")
    content = event.get("content")

    if event_type in ("session.instructions.append", "session.thinking.append"):
        if isinstance(content, str) and content:
            session.pending_context.append(content)
    elif event_type == "session.commentary.append":
        if isinstance(content, str) and content:
            delegation_id = event.get("delegation_id")
            if delegation_id in session.pending_delegations:
                session.pending_delegations.discard(delegation_id)
            _cancel_current_response(session)
            session.response_task = asyncio.ensure_future(speak_commentary(session, content))
    elif event_type == "session.input_audio.mute":
        session.muted = True
    elif event_type == "session.input_audio.unmute":
        session.muted = False
    elif event_type == "session.close":
        asyncio.ensure_future(_teardown_session(session, send_closed_event=True))


async def _teardown_session(session: Session, send_closed_event: bool) -> None:
    _cancel_current_response(session)
    if session.usage_task is not None:
        session.usage_task.cancel()
    _active_sessions.pop(session.id, None)
    if send_closed_event:
        session.send(events.usage_updated(session.usage_seconds(), closed=True))
    if session.pc is not None:
        try:
            await session.pc.close()
        except Exception:
            pass


async def _broadcast_usage(session: Session) -> None:
    try:
        while True:
            await asyncio.sleep(USAGE_BROADCAST_INTERVAL_SECONDS)
            session.send(events.usage_updated(session.usage_seconds()))
    except asyncio.CancelledError:
        pass
```

Update `create_live_session` to register the session, wire the message
handler, start the usage broadcast, and clean up on connection failure:

```python
async def create_live_session(body: LiveSessionBody) -> dict:
    offer_sdp = body.transport["sdp"]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    session = Session(pc)
    session.instructions = body.session.get("instructions", "") or ""
    _active_sessions[session.id] = session
    pc.addTrack(session.output_track)
    input_pipeline = build_input_pipeline(session)

    @pc.on("datachannel")
    def on_datachannel(channel):
        session.channel = channel
        session.send(events.session_created(session.id))
        session.send(events.session_started())

        @channel.on("message")
        def on_message(message):
            _handle_client_message(session, message)

        session.usage_task = asyncio.ensure_future(_broadcast_usage(session))

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.ensure_future(_consume_audio(track, input_pipeline, session))

    @pc.on("connectionstatechange")
    def on_connectionstatechange():
        if pc.connectionState in ("failed", "closed"):
            asyncio.ensure_future(_teardown_session(session, send_closed_event=False))

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    deadline = time.monotonic() + 10
    while pc.iceGatheringState != "complete" and time.monotonic() < deadline:
        await asyncio.sleep(0.05)

    return {
        "session": {"id": session.id},
        "transport": {"type": "webrtc", "sdp": pc.localDescription.sdp},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/serverlocal/test_session.py -v`
Expected: PASS (every test in the file — the full core-pipeline suite
plus every test this plan added across Tasks 6 and 7)

Also run the full suite to confirm no cross-file regressions:
Run: `.venv/bin/python3 -m pytest tests/serverlocal/ -v`
Expected: PASS, all files

- [ ] **Step 5: Self-review**

Before committing, confirm:
- `events.usage_updated(..., closed=True)` (used by `_teardown_session`)
  produces `{"type": "session.closed", ...}` — re-check
  `serverlocal/events.py`'s existing `usage_updated` function signature
  if you're unsure; this plan doesn't modify that function, it already
  exists from the core pipeline plan.
- `_active_sessions` is only ever mutated from `create_live_session`
  (insert) and `_teardown_session` (remove) — no other new code path
  should touch it.
- The `connectionstatechange` handler intentionally does NOT fire on
  `"disconnected"` (only `"failed"`/`"closed"`) — a transient network
  blip shouldn't tear down a session that might recover.

- [ ] **Step 6: Commit**

```bash
git add serverlocal/session.py tests/serverlocal/test_session.py
git commit -m "Handle client data-channel messages: context injection, mute, commentary, close; add session lifecycle cleanup and periodic usage broadcast"
```

---

## Self-Review Notes

- **Spec coverage:** Barge-in (Tasks 1, 6), delegation marker detection +
  `session.delegation.created` (Tasks 2, 6), client `session.commentary
  .append` → spoken output (Task 6), `session.instructions.append`/
  `session.thinking.append` → next-turn context (Task 7),
  mute/unmute (Task 7), `session.close` → `session.closed` + connection
  teardown (Task 7), periodic `session.usage.updated` (Task 7),
  `/responses` with JSON-schema + one-call web search (Tasks 3, 4, 5) —
  every piece of the spec's "Plan B" scope from
  `docs/superpowers/specs/2026-09-15-local-voice-server-design.md` has a
  task. Not covered by design (see spec's own "Open implementation-time
  details" and this plan's scope): reconnection/session resumption after
  a dropped connection, and persisted (cross-restart) session history —
  neither was ever in scope for this local dev server.
- **Type consistency checked:** `InputPipeline.current_speech_run_ms()`
  (Task 1) is called exactly that way in Task 6's `_consume_audio`
  update. `strip_delegation_markers(token_stream, on_delegation)`'s
  argument order (Task 2) matches its call site in Task 6's
  `run_response_turn`. `ollama_responses.generate`'s parameter order
  (`instructions, user_text, schema, want_search, model`) (Task 4) matches
  both its test assertions and Task 5's `responses_api.py` call site
  exactly. `Session.pending_context`/`muted`/`usage_task` (Task 7) are
  additive to `Session.response_task`/`pending_delegations` (Task 6) —
  Task 7's `__init__` block explicitly shows all five lines together to
  avoid ambiguity about which task's attributes an implementer working
  from Task 7's brief alone should expect to already exist.
- **No placeholders:** every step has complete, real code and either a
  concrete pytest command with an expected result or (Task 5's manual
  import check, matching the core pipeline plan's own precedent) a
  concrete command whose success is itself the verification.
