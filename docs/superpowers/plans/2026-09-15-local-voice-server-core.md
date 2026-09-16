# Local Voice Server — Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `serverlocal/server.py`'s plain-WebSocket turn loop with an
`aiortc`-based WebRTC server that speaks the exact same `POST /live/sessions`
+ `oai-events` data-channel protocol the Android app already implements, so
a real device on the same LAN can hold a full voice conversation against a
fully local Whisper → Ollama → Kokoro pipeline. No barge-in, no delegation,
no `/responses` yet — those are Plan B and Plan C.

**Architecture:** A FastAPI app hosts one route, `POST /live/sessions`,
that negotiates an `aiortc.RTCPeerConnection` per caller (LAN-only, no
ICE servers). Small, independently-testable modules handle each concern —
turn detection (`vad.py`), transcription/synthesis wiring (`models.py`),
sentence-chunked reply generation (`response_pipeline.py`), and the
outgoing audio track (`audio_track.py`) — and `session.py` wires them
together behind the route.

**Tech Stack:** `fastapi`, `uvicorn`, `aiortc`, `av`, `webrtcvad`,
`faster-whisper`, `kokoro-onnx`, `onnxruntime`, `huggingface_hub`,
`numpy`, `requests` (Ollama HTTP), `pytest`, `pytest-mock` (via
`unittest.mock`/`monkeypatch`, already in stdlib — no new test dep needed).

**Spec:** `docs/superpowers/specs/2026-09-15-local-voice-server-design.md`

## Global Constraints

- LAN-only: `RTCPeerConnection` configured with `iceServers=[]`, no
  STUN/TURN. (Spec: "Non-goals".)
- The **client** creates the `oai-events` data channel (see
  `LiveTransport.kt:276`); the server must attach via
  `@pc.on("datachannel")`, never call `pc.createDataChannel()` itself.
- Event field names are exact and must match `MuralViewModel.kt` byte for
  byte: `type`, `delta`, `start_ms`, `end_ms`, `event_id`,
  `session.id`, `usage.seconds`. (Spec: "Protocol compatibility".)
- `session.started` must be sent as soon as the data channel opens — the
  client only sends its first `session.instructions.append` (greeting)
  *after* receiving `session.started` (`MuralViewModel.kt:738-741`). Do
  not wait for a client message before sending it.
- Turn-boundary VAD: 20ms frames, 16kHz mono 16-bit PCM, trailing silence
  ≥500ms after speech closes a turn. (Spec: "InputPipeline".)
- Sentence-boundary characters for reply chunking: `.`, `!`, `?`, newline.
  (Spec: "ResponsePipeline".)
- Ollama at `http://localhost:11434`, model `qwen2.5:7b` (matches the
  model already referenced in the existing `serverlocal/server.py`).
- Whisper model stays `"small"`, `device="cpu"`, `compute_type="int8"` —
  unchanged from the existing file; performance tuning is out of scope
  here (YAGNI).
- Model loading (`faster-whisper` + Kokoro ONNX + voice files) must not
  run at import time — move it behind an explicit `load_models()` called
  from FastAPI's startup hook, so every other module can be imported and
  unit-tested without downloading multi-GB weights or a working GPU/CPU
  ML runtime.

---

## File Structure

```
serverlocal/
  __init__.py            # new, empty — makes serverlocal a package
  server.py               # rewritten: thin FastAPI app + uvicorn entrypoint
  models.py                # new — Whisper/Kokoro loading, extracted from
                            #   today's server.py almost verbatim, behind
                            #   load_models()
  events.py                # new — builds the exact event-vocabulary dicts
  vad.py                    # new — TurnDetector (WebRTC VAD turn-boundary)
  audio_track.py            # new — OutputAudioTrack (aiortc MediaStreamTrack)
  ollama_client.py          # new — streaming chat wrapper around Ollama
  response_pipeline.py      # new — chunk_sentences() + ResponsePipeline
  input_pipeline.py         # new — InputPipeline (frames -> turns -> text)
  session.py                # new — Session, SessionManager route, wiring
  requirements.txt          # updated with aiortc/av/webrtcvad
tests/
  serverlocal/
    __init__.py
    test_events.py
    test_vad.py
    test_audio_track.py
    test_ollama_client.py
    test_response_pipeline.py
    test_input_pipeline.py
    test_session.py
```

---

### Task 1: Package scaffolding + move model loading behind `load_models()`

**Files:**
- Create: `serverlocal/__init__.py` (empty)
- Create: `serverlocal/models.py`
- Modify: `serverlocal/server.py` (full rewrite)
- Modify: `serverlocal/requirements.txt` (create if it doesn't exist yet)
- Test: manual (see Step 5) — this task moves heavy ML setup, which can't
  be exercised by a fast automated test without real model downloads.

**Interfaces:**
- Produces: `models.load_models() -> None` (populates module globals),
  `models.transcribe(audio: np.ndarray) -> str`,
  `models.synthesize(text: str, voice: str, lang: str, speed: float = 1.0) -> tuple[np.ndarray, int]`.

- [ ] **Step 1: Create `serverlocal/__init__.py`**

Empty file. This makes `serverlocal` importable as `serverlocal.xxx` from
both `server.py` (via relative imports) and the test suite.

- [ ] **Step 2: Write `serverlocal/models.py`**

Move the existing model-loading code (current `serverlocal/server.py`
lines 1-101) into this file almost unchanged, but wrapped in a
`load_models()` function instead of running at import time, and add the
two thin wrapper functions other modules will call:

```python
import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from kokoro_onnx.session import create_session
from kokoro_onnx.tokenizer import Tokenizer
from huggingface_hub import hf_hub_download
import torch

stt_model: WhisperModel | None = None
tts_model: Kokoro | None = None


def load_models() -> None:
    global stt_model, tts_model
    if stt_model is not None and tts_model is not None:
        return

    print("Cargando Whisper (STT)...")
    stt_model = WhisperModel("small", device="cpu", compute_type="int8")

    print("Cargando Kokoro (TTS)...")
    onnx_path = hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="onnx/model.onnx")

    print("Descargando voces...")
    voices_files = {
        "af_bella": hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="voices/af_bella.bin"),
        "am_adam": hf_hub_download(repo_id="onnx-community/Kokoro-82M-ONNX", filename="voices/am_adam.bin"),
        "ef_dora": hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/ef_dora.pt"),
        "em_alex": hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/em_alex.pt"),
    }

    def prepare_voice(path):
        if path.endswith(".pt"):
            data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                data = data.get("weight", next(iter(data.values())))
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            raw = data.astype(np.float32).flatten()
        else:
            raw = np.fromfile(path, dtype=np.float32)
        if raw.size >= 256:
            return raw[:256].reshape(1, 256)
        return np.atleast_2d(raw)

    session = create_session(onnx_path)
    original_run = session.run

    def safe_run(output_names, input_feed, run_options=None):
        fixed_feed = {}
        for k, v in input_feed.items():
            arr = np.array(v)
            if k == "style" or "style" in k:
                if arr.ndim == 1:
                    arr = np.expand_dims(arr, axis=0)
                arr = arr[:, :256]
            elif arr.ndim == 1 and k != "speed":
                arr = np.expand_dims(arr, axis=0)
            fixed_feed[k] = arr
        return original_run(output_names, fixed_feed, run_options)

    session.run = safe_run

    model = Kokoro.__new__(Kokoro)
    model.session = session
    model.sess = session
    model.has_timings = False
    model.espeak_config = None
    model.voices = {name: prepare_voice(path) for name, path in voices_files.items()}
    model.tokenizer = Tokenizer()

    def _get_voice_override(self, name, *args, **kwargs):
        return self.voices.get(name, list(self.voices.values())[0])

    model.get_voice = _get_voice_override.__get__(model, Kokoro)
    if hasattr(model, "_get_voice"):
        model._get_voice = _get_voice_override.__get__(model, Kokoro)

    ONNX_TO_NUMPY_DTYPE = {
        "tensor(int64)": np.dtype("int64"),
        "tensor(int32)": np.dtype("int32"),
        "tensor(float)": np.dtype("float32"),
        "tensor(float16)": np.dtype("float16"),
        "tensor(double)": np.dtype("float64"),
    }
    inputs = session.get_inputs()
    model._tokens_input = "tokens" if any(i.name == "tokens" for i in inputs) else inputs[0].name
    model._style_input = "style" if any(i.name == "style" for i in inputs) else inputs[1].name
    model._speed_input = "speed" if any(i.name == "speed" for i in inputs) else (inputs[2].name if len(inputs) > 2 else None)
    model._input_dtypes = {i.name: ONNX_TO_NUMPY_DTYPE.get(i.type, np.dtype("int64")) for i in inputs}

    tts_model = model
    print("Modelos listos.")


def transcribe(audio: np.ndarray) -> str:
    """audio: float32 PCM in [-1, 1], 16kHz mono."""
    assert stt_model is not None, "load_models() must run before transcribe()"
    segments, _info = stt_model.transcribe(audio)
    return " ".join(segment.text for segment in segments).strip()


def synthesize(text: str, voice: str, lang: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
    assert tts_model is not None, "load_models() must run before synthesize()"
    samples, sample_rate = tts_model.create(text, voice=voice, speed=speed, lang=lang)
    return samples, sample_rate
```

- [ ] **Step 3: Rewrite `serverlocal/server.py`**

```python
from fastapi import FastAPI
import uvicorn

from . import models
from .session import router

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
async def _startup() -> None:
    models.load_models()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

(This will fail to import until Task 8 creates `session.py` — that's
expected; come back and verify this file once Task 8 is done. Steps 4-5
below only need `models.py` to exist.)

- [ ] **Step 4: Update `serverlocal/requirements.txt`**

```
fastapi
uvicorn[standard]
websockets
faster-whisper
kokoro-onnx
onnxruntime
huggingface_hub
soundfile
numpy
torch
requests
aiortc
av
webrtcvad
```

- [ ] **Step 5: Manual verification (models load correctly)**

This can't be automated in the test suite (multi-GB downloads, real
CPU inference). Run by hand once, on the Mac that will host the server:

```bash
cd serverlocal && pip install -r requirements.txt && python3 -c "from serverlocal import models; models.load_models(); print(models.transcribe.__doc__ or 'ok')"
```

Expected: prints `Cargando Whisper (STT)...`, `Cargando Kokoro (TTS)...`,
`Descargando voces...`, `Modelos listos.` with no traceback.

- [ ] **Step 6: Commit**

```bash
git add serverlocal/__init__.py serverlocal/models.py serverlocal/server.py serverlocal/requirements.txt
git commit -m "Extract model loading into serverlocal/models.py behind load_models()"
```

---

### Task 2: `events.py` — protocol event builders

**Files:**
- Create: `serverlocal/events.py`
- Test: `tests/serverlocal/test_events.py`

**Interfaces:**
- Produces: `session_created(session_id)`, `session_started()`,
  `transcript_delta(kind, text, start_ms, end_ms)`,
  `usage_updated(seconds, closed=False)`, `error(message)` — each
  returns a `dict` ready for `json.dumps`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_events.py
from serverlocal import events


def test_session_created_shape():
    event = events.session_created("abc-123")
    assert event == {"type": "mural.session.created", "session": {"id": "abc-123"}}


def test_session_started_shape():
    assert events.session_started() == {"type": "session.started"}


def test_transcript_delta_input():
    event = events.transcript_delta("input", "hola", 100, 400)
    assert event["type"] == "session.input_transcript.delta"
    assert event["delta"] == "hola"
    assert event["start_ms"] == 100
    assert event["end_ms"] == 400
    assert isinstance(event["event_id"], str) and event["event_id"]


def test_transcript_delta_output():
    event = events.transcript_delta("output", "hi", 0, 200)
    assert event["type"] == "session.output_transcript.delta"


def test_transcript_delta_rejects_bad_kind():
    try:
        events.transcript_delta("bogus", "x", 0, 1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_usage_updated_shape():
    assert events.usage_updated(12.5) == {"type": "session.usage.updated", "usage": {"seconds": 12.5}}


def test_usage_updated_closed_shape():
    assert events.usage_updated(30.0, closed=True) == {"type": "session.closed", "usage": {"seconds": 30.0}}


def test_error_shape():
    assert events.error("bad thing") == {"type": "error", "message": "bad thing"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/serverlocal/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serverlocal.events'`

- [ ] **Step 3: Write `serverlocal/events.py`**

```python
import uuid


def session_created(session_id: str) -> dict:
    return {"type": "mural.session.created", "session": {"id": session_id}}


def session_started() -> dict:
    return {"type": "session.started"}


def transcript_delta(kind: str, text: str, start_ms: int, end_ms: int) -> dict:
    if kind not in ("input", "output"):
        raise ValueError(f"kind must be 'input' or 'output', got {kind!r}")
    return {
        "type": f"session.{kind}_transcript.delta",
        "delta": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "event_id": str(uuid.uuid4()),
    }


def usage_updated(seconds: float, closed: bool = False) -> dict:
    return {
        "type": "session.closed" if closed else "session.usage.updated",
        "usage": {"seconds": seconds},
    }


def error(message: str) -> dict:
    return {"type": "error", "message": message}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/serverlocal/test_events.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add serverlocal/events.py tests/serverlocal/test_events.py
git commit -m "Add protocol event builders matching MuralViewModel's event vocabulary"
```

---

### Task 3: `vad.py` — turn-boundary detection

**Files:**
- Create: `serverlocal/vad.py`
- Test: `tests/serverlocal/test_vad.py`

**Interfaces:**
- Produces: `TurnDetector` with `.feed(pcm_bytes: bytes) -> bytes | None`
  and class constants `FRAME_BYTES`, `FRAME_MS`, `SILENCE_MS_TO_CLOSE`.
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_vad.py
import webrtcvad

from serverlocal.vad import TurnDetector


def _frame_bytes():
    return b"\x00\x00" * (TurnDetector.FRAME_BYTES // 2)


def test_closes_turn_after_500ms_trailing_silence(monkeypatch):
    # 3 "speech" frames, then 25 "silence" frames (25 * 20ms = 500ms).
    pattern = [True, True, True] + [False] * 25
    calls = iter(pattern)
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: next(calls))

    detector = TurnDetector()
    closed = None
    for _ in range(len(pattern)):
        result = detector.feed(_frame_bytes())
        if result is not None:
            closed = result

    assert closed is not None
    assert len(closed) == len(pattern) * TurnDetector.FRAME_BYTES


def test_does_not_close_before_silence_threshold(monkeypatch):
    # 3 speech frames, then only 20 silence frames (400ms) -- not enough.
    pattern = [True, True, True] + [False] * 20
    calls = iter(pattern)
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: next(calls))

    detector = TurnDetector()
    closed = None
    for _ in range(len(pattern)):
        result = detector.feed(_frame_bytes())
        if result is not None:
            closed = result

    assert closed is None


def test_pure_silence_never_closes_a_turn(monkeypatch):
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: False)

    detector = TurnDetector()
    closed = None
    for _ in range(50):
        result = detector.feed(_frame_bytes())
        if result is not None:
            closed = result

    assert closed is None


def test_buffers_partial_frames_across_feed_calls(monkeypatch):
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: False)

    detector = TurnDetector()
    whole_frame = _frame_bytes()
    half = len(whole_frame) // 2
    # Feed the same frame's worth of bytes split across two calls.
    result_a = detector.feed(whole_frame[:half])
    result_b = detector.feed(whole_frame[half:])

    assert result_a is None
    assert result_b is None
    assert len(detector._buffer) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/serverlocal/test_vad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serverlocal.vad'`

- [ ] **Step 3: Write `serverlocal/vad.py`**

```python
import webrtcvad


class TurnDetector:
    """Buffers 16kHz mono 16-bit PCM, classifies 20ms frames with WebRTC
    VAD, and reports a closed turn once >=500ms of trailing silence
    follows speech."""

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

    def feed(self, pcm_bytes: bytes) -> bytes | None:
        self._buffer += pcm_bytes
        closed_turn = None
        while len(self._buffer) >= self.FRAME_BYTES:
            frame = bytes(self._buffer[: self.FRAME_BYTES])
            del self._buffer[: self.FRAME_BYTES]
            is_speech = self._vad.is_speech(frame, self.SAMPLE_RATE)
            if is_speech:
                self._in_speech = True
                self._silence_ms = 0
                self._turn_audio += frame
            elif self._in_speech:
                self._turn_audio += frame
                self._silence_ms += self.FRAME_MS
                if self._silence_ms >= self.SILENCE_MS_TO_CLOSE:
                    closed_turn = bytes(self._turn_audio)
                    self._turn_audio = bytearray()
                    self._in_speech = False
                    self._silence_ms = 0
        return closed_turn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/serverlocal/test_vad.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add serverlocal/vad.py tests/serverlocal/test_vad.py
git commit -m "Add WebRTC-VAD-based turn boundary detector"
```

---

### Task 4: `audio_track.py` — outgoing WebRTC audio track

**Files:**
- Create: `serverlocal/audio_track.py`
- Test: `tests/serverlocal/test_audio_track.py`

**Interfaces:**
- Produces: `OutputAudioTrack` (subclass of `aiortc.mediastreams.MediaStreamTrack`)
  with `async push_pcm(samples: np.ndarray, sample_rate: int) -> None`,
  `clear() -> None`, `async recv() -> av.AudioFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_audio_track.py
import asyncio

import numpy as np

from serverlocal.audio_track import OutputAudioTrack


def test_recv_returns_frame_with_correct_rate_and_increasing_pts():
    track = OutputAudioTrack()
    samples = np.array([100, -100, 200, -200], dtype=np.int16)

    async def scenario():
        await track.push_pcm(samples, sample_rate=24000)
        frame1 = await track.recv()
        await track.push_pcm(samples, sample_rate=24000)
        frame2 = await track.recv()
        return frame1, frame2

    frame1, frame2 = asyncio.run(scenario())
    assert frame1.sample_rate == 24000
    assert frame1.samples == 4
    assert frame1.pts == 0
    assert frame2.pts == 4


def test_clear_drops_queued_but_unsent_frames():
    track = OutputAudioTrack()
    samples = np.zeros(10, dtype=np.int16)

    async def scenario():
        await track.push_pcm(samples, sample_rate=16000)
        await track.push_pcm(samples, sample_rate=16000)
        track.clear()
        return track._queue.qsize()

    assert asyncio.run(scenario()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/serverlocal/test_audio_track.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serverlocal.audio_track'`

- [ ] **Step 3: Write `serverlocal/audio_track.py`**

```python
import asyncio
import fractions

import av
import numpy as np
from aiortc.mediastreams import MediaStreamTrack


class OutputAudioTrack(MediaStreamTrack):
    """Outgoing WebRTC audio track fed by pushing PCM chunks as they're
    synthesized. aiortc's Opus encoder resamples whatever format/rate it's
    given to 48kHz stereo internally, so this track passes through
    Kokoro's native mono output untouched."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._timestamp = 0

    async def push_pcm(self, samples: np.ndarray, sample_rate: int) -> None:
        frame = av.AudioFrame.from_ndarray(
            samples.astype(np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = sample_rate
        await self._queue.put(frame)

    def clear(self) -> None:
        """Drops queued-but-unsent audio. A no-op if nothing is queued."""
        while not self._queue.empty():
            self._queue.get_nowait()

    async def recv(self) -> av.AudioFrame:
        frame = await self._queue.get()
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, frame.sample_rate)
        self._timestamp += frame.samples
        return frame
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/serverlocal/test_audio_track.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add serverlocal/audio_track.py tests/serverlocal/test_audio_track.py
git commit -m "Add OutputAudioTrack for streaming synthesized PCM over WebRTC"
```

---

### Task 5: `ollama_client.py` — streaming reply wrapper

**Files:**
- Create: `serverlocal/ollama_client.py`
- Test: `tests/serverlocal/test_ollama_client.py`

**Interfaces:**
- Produces: `stream_reply(prompt: str, model: str = "qwen2.5:7b") -> Iterator[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_ollama_client.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/serverlocal/test_ollama_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serverlocal.ollama_client'`

- [ ] **Step 3: Write `serverlocal/ollama_client.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/serverlocal/test_ollama_client.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add serverlocal/ollama_client.py tests/serverlocal/test_ollama_client.py
git commit -m "Add streaming Ollama client wrapper"
```

---

### Task 6: `response_pipeline.py` — sentence chunking + reply pipeline

**Files:**
- Create: `serverlocal/response_pipeline.py`
- Test: `tests/serverlocal/test_response_pipeline.py`

**Interfaces:**
- Consumes: nothing concrete from earlier tasks (takes injected callables
  so it stays independently testable).
- Produces: `chunk_sentences(token_stream: Iterable[str]) -> Iterator[str]`,
  `ResponsePipeline(stream_reply_fn, synthesize_fn, push_audio_fn, on_output_delta)`
  with `async run(prompt: str) -> str`. Later tasks (`session.py`) supply:
  - `stream_reply_fn(prompt: str) -> Iterable[str]`
  - `synthesize_fn(text: str) -> tuple[np.ndarray, int]`
  - `push_audio_fn(samples: np.ndarray, sample_rate: int) -> Awaitable[None]`
  - `on_output_delta(text: str, start_ms: int, end_ms: int) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_response_pipeline.py
import asyncio

import numpy as np

from serverlocal.response_pipeline import ResponsePipeline, chunk_sentences


def test_chunk_sentences_splits_on_terminators():
    tokens = ["Hello", " world.", " How are", " you?", " Extra"]
    assert list(chunk_sentences(tokens)) == ["Hello world.", "How are you?", "Extra"]


def test_chunk_sentences_handles_no_terminator_at_all():
    assert list(chunk_sentences(["just", " text", " no end"])) == ["just text no end"]


def test_chunk_sentences_handles_empty_stream():
    assert list(chunk_sentences([])) == []


def test_response_pipeline_streams_sentences_to_audio_and_deltas():
    deltas = []
    pushed = []

    def stream_reply_fn(prompt):
        assert prompt == "hola"
        return iter(["Hello", " world.", " Bye."])

    def synthesize_fn(text):
        return np.zeros(1600, dtype=np.int16), 16000  # 100ms @ 16kHz

    async def push_audio_fn(samples, sample_rate):
        pushed.append((len(samples), sample_rate))

    def on_output_delta(text, start_ms, end_ms):
        deltas.append((text, start_ms, end_ms))

    pipeline = ResponsePipeline(stream_reply_fn, synthesize_fn, push_audio_fn, on_output_delta)
    full_text = asyncio.run(pipeline.run("hola"))

    assert full_text == "Hello world. Bye."
    assert deltas == [("Hello world.", 0, 100), ("Bye.", 100, 200)]
    assert pushed == [(1600, 16000), (1600, 16000)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/serverlocal/test_response_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serverlocal.response_pipeline'`

- [ ] **Step 3: Write `serverlocal/response_pipeline.py`**

```python
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator

import numpy as np

SENTENCE_END = re.compile(r"[.!?\n]")


def chunk_sentences(token_stream: Iterable[str]) -> Iterator[str]:
    """Consumes an iterable of text chunks, yields complete sentences as
    they're formed. Any trailing text with no terminator is yielded once
    the stream ends."""
    buffer = ""
    for token in token_stream:
        buffer += token
        while True:
            match = SENTENCE_END.search(buffer)
            if not match:
                break
            end = match.end()
            sentence = buffer[:end].strip()
            buffer = buffer[end:]
            if sentence:
                yield sentence
    remainder = buffer.strip()
    if remainder:
        yield remainder


class ResponsePipeline:
    """Turns a user turn into streamed spoken output: LLM tokens -> sentence
    chunks -> TTS -> pushed onto the outgoing audio track, emitting one
    output_transcript.delta per sentence."""

    def __init__(
        self,
        stream_reply_fn: Callable[[str], Iterable[str]],
        synthesize_fn: Callable[[str], tuple[np.ndarray, int]],
        push_audio_fn: Callable[[np.ndarray, int], Awaitable[None]],
        on_output_delta: Callable[[str, int, int], None],
    ):
        self._stream_reply = stream_reply_fn
        self._synthesize = synthesize_fn
        self._push_audio = push_audio_fn
        self._on_delta = on_output_delta

    async def run(self, prompt: str) -> str:
        """Runs one full turn's reply and returns the full reply text.
        A caller may wrap this in an asyncio.Task and cancel it for
        barge-in (added in a later plan)."""
        elapsed_ms = 0
        parts: list[str] = []
        for sentence in chunk_sentences(self._stream_reply(prompt)):
            parts.append(sentence)
            samples, sample_rate = self._synthesize(sentence)
            await self._push_audio(samples, sample_rate)
            duration_ms = int(len(samples) / sample_rate * 1000)
            self._on_delta(sentence, elapsed_ms, elapsed_ms + duration_ms)
            elapsed_ms += duration_ms
        return " ".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/serverlocal/test_response_pipeline.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add serverlocal/response_pipeline.py tests/serverlocal/test_response_pipeline.py
git commit -m "Add sentence-chunked streaming response pipeline"
```

---

### Task 7: `input_pipeline.py` — frames to transcribed turns

**Files:**
- Create: `serverlocal/input_pipeline.py`
- Test: `tests/serverlocal/test_input_pipeline.py`

**Interfaces:**
- Consumes: `TurnDetector` from Task 3 (`serverlocal.vad`).
- Produces: `InputPipeline(transcribe_fn, on_turn_transcribed)` with
  `handle_frame(frame: av.AudioFrame) -> None`. Later tasks supply:
  - `transcribe_fn(audio: np.ndarray) -> str`
  - `on_turn_transcribed(text: str, start_ms: int, end_ms: int) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_input_pipeline.py
import av
import numpy as np
import webrtcvad

from serverlocal.input_pipeline import InputPipeline
from serverlocal.vad import TurnDetector


def _silence_frame(num_samples=320, sample_rate=16000):
    samples = np.zeros((1, num_samples), dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
    frame.sample_rate = sample_rate
    return frame


def test_handle_frame_transcribes_closed_turn_and_reports_it(monkeypatch):
    # 3 "speech" frames then 25 "silence" frames closes exactly one turn.
    pattern = [True, True, True] + [False] * 25
    calls = iter(pattern)
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: next(calls))

    captured_audio = {}

    def fake_transcribe(audio_np):
        captured_audio["array"] = audio_np
        return "hola de prueba"

    turns = []

    def on_turn_transcribed(text, start_ms, end_ms):
        turns.append((text, start_ms, end_ms))

    pipeline = InputPipeline(fake_transcribe, on_turn_transcribed)
    for _ in range(len(pattern)):
        pipeline.handle_frame(_silence_frame())

    assert turns == [("hola de prueba", 0, len(pattern) * 20)]
    assert captured_audio["array"].dtype == np.float32
    assert captured_audio["array"].max() <= 1.0
    assert captured_audio["array"].min() >= -1.0


def test_handle_frame_does_not_call_transcribe_without_a_closed_turn(monkeypatch):
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: False)

    calls = []
    pipeline = InputPipeline(lambda a: calls.append(a) or "x", lambda *a: None)
    for _ in range(10):
        pipeline.handle_frame(_silence_frame())

    assert calls == []


def test_handle_frame_skips_on_turn_callback_for_blank_transcription(monkeypatch):
    pattern = [True, True, True] + [False] * 25
    calls = iter(pattern)
    monkeypatch.setattr(webrtcvad.Vad, "is_speech", lambda self, frame, rate: next(calls))

    turns = []
    pipeline = InputPipeline(lambda a: "   ", lambda *a: turns.append(a))
    for _ in range(len(pattern)):
        pipeline.handle_frame(_silence_frame())

    assert turns == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/serverlocal/test_input_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serverlocal.input_pipeline'`

- [ ] **Step 3: Write `serverlocal/input_pipeline.py`**

```python
from collections.abc import Callable

import av
import numpy as np

from .vad import TurnDetector


class InputPipeline:
    """Consumes incoming client audio frames, detects turn boundaries via
    TurnDetector, and hands off each closed turn's audio to a
    transcription callback."""

    def __init__(
        self,
        transcribe_fn: Callable[[np.ndarray], str],
        on_turn_transcribed: Callable[[str, int, int], None],
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
                text = self._transcribe(audio_np).strip()
                if text:
                    self._on_turn(text, start_ms, end_ms)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/serverlocal/test_input_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add serverlocal/input_pipeline.py tests/serverlocal/test_input_pipeline.py
git commit -m "Add InputPipeline: audio frames to transcribed turns"
```

---

### Task 8: `session.py` — wire it all behind `POST /live/sessions`

**Files:**
- Create: `serverlocal/session.py`
- Test: `tests/serverlocal/test_session.py`

**Interfaces:**
- Consumes: `events` (Task 2), `OutputAudioTrack` (Task 4),
  `ollama_client.stream_reply` (Task 5), `ResponsePipeline` (Task 6),
  `InputPipeline` (Task 7), `models.transcribe`/`models.synthesize`
  (Task 1).
- Produces: FastAPI `router` with `POST /live/sessions`; `Session`,
  `build_input_pipeline(session)`, `run_response_turn(session, user_text)`
  — the last two are what Task 8's own tests exercise directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/serverlocal/test_session.py
import asyncio

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription

from serverlocal import events, models, ollama_client, session as session_module
from serverlocal.session import LiveSessionBody, Session, build_input_pipeline, create_live_session, run_response_turn


def test_create_live_session_returns_a_valid_answer():
    async def scenario():
        client_pc = RTCPeerConnection()
        client_pc.createDataChannel("oai-events")
        offer = await client_pc.createOffer()
        await client_pc.setLocalDescription(offer)

        body = LiveSessionBody(session={}, transport={"type": "webrtc", "sdp": client_pc.localDescription.sdp})
        result = await create_live_session(body)

        assert isinstance(result["session"]["id"], str) and result["session"]["id"]
        assert result["transport"]["type"] == "webrtc"
        assert result["transport"]["sdp"].startswith("v=0")

        # A malformed answer would raise here.
        await client_pc.setRemoteDescription(RTCSessionDescription(sdp=result["transport"]["sdp"], type="answer"))
        await client_pc.close()

    asyncio.run(scenario())


def test_session_emits_created_then_started_when_channel_opens():
    async def scenario():
        client_pc = RTCPeerConnection()
        channel = client_pc.createDataChannel("oai-events")
        received = []
        opened = asyncio.Event()

        @channel.on("open")
        def _on_open():
            opened.set()

        @channel.on("message")
        def _on_message(message):
            import json

            received.append(json.loads(message))

        offer = await client_pc.createOffer()
        await client_pc.setLocalDescription(offer)
        body = LiveSessionBody(session={}, transport={"type": "webrtc", "sdp": client_pc.localDescription.sdp})
        result = await create_live_session(body)
        await client_pc.setRemoteDescription(RTCSessionDescription(sdp=result["transport"]["sdp"], type="answer"))

        await asyncio.wait_for(opened.wait(), timeout=5)
        # Give the two just-sent events a moment to arrive.
        for _ in range(50):
            if len(received) >= 2:
                break
            await asyncio.sleep(0.1)

        assert received[0]["type"] == "mural.session.created"
        assert received[0]["session"]["id"] == result["session"]["id"]
        assert received[1]["type"] == "session.started"
        await client_pc.close()

    asyncio.run(scenario())


def test_on_turn_transcribed_sends_input_delta_immediately(monkeypatch):
    sent = []
    session = Session(pc=None)
    session.send = sent.append  # type: ignore[assignment]
    monkeypatch.setattr(session_module.asyncio, "ensure_future", lambda coro: coro.close())

    pipeline = build_input_pipeline(session)
    pipeline._on_turn("hola de prueba", 0, 500)

    assert len(sent) == 1
    assert sent[0]["type"] == "session.input_transcript.delta"
    assert sent[0]["delta"] == "hola de prueba"


def test_run_response_turn_sends_output_deltas_and_pushes_audio(monkeypatch):
    monkeypatch.setattr(ollama_client, "stream_reply", lambda prompt: iter(["Hola", " mundo."]))
    monkeypatch.setattr(models, "synthesize", lambda text, voice, lang: (np.zeros(1600, dtype=np.int16), 16000))

    pushed = []
    sent = []
    session = Session(pc=None)
    session.send = sent.append  # type: ignore[assignment]

    class FakeTrack:
        async def push_pcm(self, samples, sample_rate):
            pushed.append((len(samples), sample_rate))

    session.output_track = FakeTrack()  # type: ignore[assignment]

    asyncio.run(run_response_turn(session, "hola"))

    assert pushed == [(1600, 16000)]
    assert [event["type"] for event in sent] == ["session.output_transcript.delta"]
    assert sent[0]["delta"] == "Hola mundo."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/serverlocal/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'serverlocal.session'`

- [ ] **Step 3: Write `serverlocal/session.py`**

```python
import asyncio
import json
import time
import uuid

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter
from pydantic import BaseModel

from . import events, models, ollama_client
from .audio_track import OutputAudioTrack
from .input_pipeline import InputPipeline
from .response_pipeline import ResponsePipeline

router = APIRouter()


class LiveSessionBody(BaseModel):
    session: dict
    transport: dict


class Session:
    def __init__(self, pc: RTCPeerConnection | None):
        self.id = str(uuid.uuid4())
        self.pc = pc
        self.channel = None
        self.output_track = OutputAudioTrack()
        self.started_at = time.monotonic()

    def send(self, event: dict) -> None:
        if self.channel is not None and self.channel.readyState == "open":
            self.channel.send(json.dumps(event))

    def usage_seconds(self) -> float:
        return time.monotonic() - self.started_at


def build_input_pipeline(session: Session) -> InputPipeline:
    def transcribe_fn(audio_np):
        return models.transcribe(audio_np)

    def on_turn_transcribed(text: str, start_ms: int, end_ms: int) -> None:
        session.send(events.transcript_delta("input", text, start_ms, end_ms))
        asyncio.ensure_future(run_response_turn(session, text))

    return InputPipeline(transcribe_fn, on_turn_transcribed)


async def run_response_turn(session: Session, user_text: str) -> None:
    def stream_reply_fn(prompt: str):
        return ollama_client.stream_reply(prompt)

    def synthesize_fn(text: str):
        return models.synthesize(text, voice="ef_dora", lang="es")

    async def push_audio_fn(samples, sample_rate):
        await session.output_track.push_pcm(samples, sample_rate)

    def on_output_delta(text: str, start_ms: int, end_ms: int) -> None:
        session.send(events.transcript_delta("output", text, start_ms, end_ms))

    pipeline = ResponsePipeline(stream_reply_fn, synthesize_fn, push_audio_fn, on_output_delta)
    await pipeline.run(user_text)


async def create_live_session(body: LiveSessionBody) -> dict:
    offer_sdp = body.transport["sdp"]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    session = Session(pc)
    pc.addTrack(session.output_track)
    input_pipeline = build_input_pipeline(session)

    @pc.on("datachannel")
    def on_datachannel(channel):
        session.channel = channel
        session.send(events.session_created(session.id))
        session.send(events.session_started())

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.ensure_future(_consume_audio(track, input_pipeline))

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


async def _consume_audio(track, input_pipeline: InputPipeline) -> None:
    while True:
        try:
            frame = await track.recv()
        except Exception:
            break
        input_pipeline.handle_frame(frame)


@router.post("/live/sessions")
async def live_sessions_route(body: LiveSessionBody) -> dict:
    return await create_live_session(body)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/serverlocal/test_session.py -v`
Expected: 4 passed

If `test_session_emits_created_then_started_when_channel_opens` is flaky
locally (two in-process `aiortc` peer connections negotiating DTLS/SCTP
over loopback can occasionally need more than a couple hundred ms): raise
the `asyncio.wait_for` timeout and the polling loop's iteration count
before concluding something is actually broken — this is a known
characteristic of `aiortc`'s own test suite (see their
`tests/test_rtcpeerconnection.py` for the same two-peer-in-one-process
pattern), not a sign the production code is wrong.

- [ ] **Step 5: Verify `serverlocal/server.py` now imports cleanly**

Run: `python3 -c "import serverlocal.server"`
Expected: no traceback (model loading is deferred to the FastAPI startup
hook, so this succeeds without downloading anything).

- [ ] **Step 6: Commit**

```bash
git add serverlocal/session.py tests/serverlocal/test_session.py
git commit -m "Wire SessionManager: POST /live/sessions over aiortc with matching event protocol"
```

---

## Self-Review Notes

- **Spec coverage:** SessionManager (Task 8), InputPipeline+VAD (Tasks 3,
  7), ResponsePipeline+streaming TTS (Tasks 4, 5, 6), model loading reuse
  (Task 1), exact event vocabulary (Task 2) — all covered. Barge-in,
  `DelegationBridge`, and `/responses` are explicitly out of scope for
  this plan (Plan B). Usage/session-lifecycle events beyond the two
  emitted at connect time are also Plan B/C territory (periodic
  `session.usage.updated` and `session.closed` on explicit close need the
  data-channel message handler for `session.close`, which arrives once
  Plan B adds handling for the rest of the client-sent event types).
- **Type consistency checked:** `ResponsePipeline`'s `synthesize_fn`
  signature `(text) -> (np.ndarray, int)` matches `models.synthesize`'s
  return `(samples, sample_rate)` used in Task 8's wiring;
  `InputPipeline`'s `transcribe_fn` signature `(np.ndarray) -> str`
  matches `models.transcribe`; `Session.send` used identically in Tasks 8
  and its own tests.
- **No placeholders:** every step above has runnable code and a concrete
  expected result; the one manual step (Task 1, Step 5) is manual because
  it requires real multi-GB model downloads, not because anything is
  undecided.
