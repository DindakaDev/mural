# serverlocal

A local WebRTC voice server that replaces OpenAI's Realtime API for the Mural Android app: it accepts a WebRTC offer over `POST /live/sessions`, transcribes incoming microphone audio with Whisper (faster-whisper), generates replies with a local Ollama model, synthesizes speech with Kokoro TTS, and streams the reply back to the client as a real-time audio track — all on-device, with no cloud dependency.

## Requirements

- macOS (tested on macOS 26 Tahoe, Apple Silicon)
- Python 3.11 (not 3.14 — `ensurepip` is broken on 3.14)
- [uv](https://github.com/astral-sh/uv) — replaces pip (pip's truststore breaks on macOS 26)
- [Ollama](https://ollama.com) with `qwen2.5:7b` pulled
- Android device connected via USB with ADB enabled

## Install

### 1. Install uv (if not already installed)

```bash
brew install uv
```

### 2. Create virtual environment with Python 3.11

```bash
cd serverlocal
uv venv .venv --python 3.11
```

### 3. Install dependencies

```bash
uv pip install --python .venv/bin/python3.11 "setuptools<81" "cryptography<49" -r requirements.txt
```

Known constraints:
- `setuptools<81` — `webrtcvad`'s `pkg_resources` shim breaks on newer setuptools.
- `cryptography<49` — needed on some Intel/arm64 toolchains while building `aiortc`.

### 4. Install and start Ollama

```bash
brew install ollama
ollama pull qwen2.5:7b
ollama serve              # runs on localhost:11434
```

To start Ollama automatically at login:
```bash
brew services start ollama
```

## Run the server

From the repository root (not from inside `serverlocal/`):

```bash
cd /path/to/mural
serverlocal/.venv/bin/python -m uvicorn serverlocal.server:app --host 0.0.0.0 --port 8000
```

Server starts on `http://0.0.0.0:8000`. Logs print to stdout.

> Running `cd serverlocal && python server.py` breaks relative imports — always run from the repo root.

`KMP_DUPLICATE_LIB_OK=TRUE` is set automatically by `server.py` at import time.

## Build and install Android app

Connect your Android device via USB, enable USB debugging, then from `apps/android/`:

```bash
cd apps/android
./gradlew installDebug
```

This builds a debug APK and installs it directly on the connected device.

## Connect the app

Make sure the Android device and your Mac are on the same Wi-Fi network. Set the server URL in the app to:

```
http://<your-mac-ip>:8000
```

Find your local IP with:
```bash
ipconfig getifaddr en0
```
