# serverlocal

A local WebRTC voice server that replaces OpenAI's Realtime API for the Mural Android app: it accepts a WebRTC offer over `POST /live/sessions`, transcribes incoming microphone audio with Whisper (faster-whisper), generates replies with a local Ollama model, synthesizes speech with Kokoro TTS, and streams the reply back to the client as a real-time audio track — all on-device, with no cloud dependency.

## Install

```
pip install -r requirements.txt
```

If installation fails, apply these two known constraints manually (they're documented at the top of `requirements.txt` but intentionally not pinned there, to keep that file matching the original spec verbatim):

- `setuptools<81` — `webrtcvad`'s `pkg_resources` shim breaks on newer setuptools.
- `cryptography<49` — needed on some Intel/arm64 toolchains while building `aiortc`.

## Run

From the repository root (not from inside `serverlocal/`):

```
python -m serverlocal.server
```

Running `cd serverlocal && python server.py` instead will break the package's relative imports (`from . import models`, etc.) — always run it as a module from the repo root.

`KMP_DUPLICATE_LIB_OK=TRUE` is set automatically by `server.py` itself at import time, so no manual environment variable setup is needed.
