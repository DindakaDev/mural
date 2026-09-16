# Local voice/LLM server for Mural (Android) — design spec

Date: 2026-09-15
Status: approved for planning
Scope: Android only (iOS deliberately out of scope for this iteration)

## Goal

Replace OpenAI (Realtime WebRTC voice API + `/responses` text API) with a
fully local stack running on the user's own Mac (Apple Silicon) on the same
LAN as the Android device, while preserving as much of the existing feature
set as practical:

- Live voice conversation (WebRTC, live transcripts, mid-conversation
  instruction injection, barge-in/interruption)
- Delegated research/lookup during a live conversation (client-side
  "delegation" pattern already implemented in `MuralViewModel`)
- Structured lesson/quiz generation (`respond()`, strict JSON schema + one
  web-search tool call)

Stack: `fastapi`, `uvicorn[standard]`, `websockets`, `faster-whisper`,
`kokoro-onnx`, `onnxruntime`, `huggingface_hub`, `soundfile`, `numpy`,
`torch`, `requests`, plus new additions: `aiortc`, `av`, an Ollama instance
(`qwen2.5:7b` to start), `duckduckgo-search`.

## Non-goals

- No STUN/TURN, no NAT traversal, no remote (non-LAN) access. Same-LAN only,
  host ICE candidates only — matches how `LiveTransport.kt` is already
  configured (`GATHER_ONCE`, no `iceServers`).
- No billing/minutes accounting. `HOSTED_MINUTES` and `MinuteCommerceClient`
  flows are untouched and unused in local mode.
- No iOS changes in this iteration.
- No attempt to match OpenAI's model quality — accepted tradeoff for running
  fully local.

## Why WebRTC (not a WebSocket rewrite)

Considered switching the live-voice transport to plain WebSocket streaming
(matches the already-added `serverlocal/server.py` shape) instead of WebRTC.
Rejected per explicit user choice: keeping WebRTC preserves the existing
Android client's audio pipeline (echo cancellation, audio focus/routing,
metering, all in `LiveTransport.kt`) untouched, and preserves barge-in as a
first-class feature rather than something bolted onto a turn-based
request/response loop.

Because the connection stays on LAN, WebRTC's main normal justification
(NAT traversal over the public internet) doesn't matter here — what we get
to keep for free is the *client-side* audio engineering already built and
tested (`JavaAudioDeviceModule`, hardware AEC, `AudioFocusRequest`,
communication-device routing), which would otherwise need to be rebuilt for
a raw WebSocket PCM stream.

## Protocol compatibility (the load-bearing decision)

`LiveTransport.kt` and `MuralViewModel.kt` do not encode any OpenAI-specific
assumptions beyond: the SDP answer arrives via `POST live/sessions`, and a
fixed vocabulary of JSON events flows over the `oai-events` data channel.
The design keeps that vocabulary **byte-for-byte identical**, so the
Android client requires only a base-URL/auth change (see "Android client
changes"), not a rewrite.

Events the client already knows how to send and receive (confirmed from
`MuralViewModel.kt`, `LiveTransport.kt`):

Client → server (data channel):
- `session.instructions.append` `{content}` — silent context injection
  (system-prompt-like), used for greeting, theme redirects, help requests.
- `session.thinking.append` `{content}` — silent context, used for
  progress/teaching-state hints.
- `session.commentary.append` `{content, delegation_id}` — text the
  server should have the live voice actually speak, in response to a
  delegation.
- `session.input_audio.mute` / `session.input_audio.unmute`
- `session.close`

Server → client (data channel):
- `mural.session.created` `{session:{id}}`
- `session.started`
- `session.input_transcript.delta` / `session.output_transcript.delta`
  `{delta, start_ms, end_ms, event_id}`
- `session.delegation.created` `{delegation:{target:"client", id}}` — only
  `target:"client"` is handled by the app today; the app then calls
  `respond()` (the `/responses` endpoint) itself and reports back via
  `session.commentary.append`.
- `session.usage.updated` / `session.closed` `{usage:{seconds}}`
- `error`

`POST /live/sessions` request/response shape (from `LiveTransport.kt`
line ~76 and `APIClient.kt` `createLiveSession`):
```
POST live/sessions
{ "session": {"model": ..., "instructions": ..., "input": [...history],
              "store": false, "delegation": {"type": "client"},
              "audio": {"output": {"voice": ...}}},
  "transport": {"type": "webrtc", "sdp": <offer>} }
→ { "session": {"id": ...}, "transport": {"type": "webrtc", "sdp": <answer>} }
```

## Server architecture

Rewrite `serverlocal/server.py` on top of `aiortc` (keep `fastapi`/`uvicorn`
as the HTTP layer around it; a plain `websockets`/`FastAPI WebSocket` route
stays available only for a health/debug endpoint, not for audio).

Model loading (Whisper init, Kokoro ONNX session + voice patching from the
existing file) is reused as-is — that part already works and is unrelated
to transport.

### SessionManager
- Handles `POST /live/sessions`: builds an `aiortc.RTCPeerConnection` with
  no ICE servers configured (LAN-only, matches client), sets the client's
  SDP offer as remote description, adds a local `OutputAudioTrack` (below),
  creates/handles the `oai-events` data channel, waits for local ICE
  gathering to finish (non-trickle, single HTTP round trip — the client
  already gathers fully before POSTing, so the server must too), returns
  the SDP answer synchronously in the HTTP response.
- One `Session` object per connection holds: peer connection, data channel,
  conversation history, current language, references to the running
  `InputPipeline`/`ResponsePipeline` tasks, and a monotonic start time for
  `usage.seconds`.
- Emits `mural.session.created` right after the session object is built,
  then `session.started` as soon as the data channel's `readyState`
  becomes `open`. The client waits for `session.started` before sending
  its first `session.instructions.append` (the greeting) — see
  `MuralViewModel.kt:738-741` — so the server must not wait for that
  message before emitting `session.started`.

### InputPipeline (per session)
- Receives frames from the client's incoming `MediaStreamTrack` (aiortc
  delivers decoded PCM via `track.recv()`).
- Runs VAD per ~20-30ms frame (`webrtcvad`, cheapest option, no extra ML
  model to load). Accumulates speech frames; ~500ms of trailing silence
  closes the turn.
- On turn close: full-turn buffer → `faster-whisper` → text → emit
  `session.input_transcript.delta` (single delta covering the whole turn is
  acceptable; true incremental partials are not required for feature
  parity — the client only renders deltas, it doesn't require several).
- Also runs continuously *while the bot is speaking* (see Barge-in below).

### ResponsePipeline (per turn)
- Takes the turn's transcript + rolling history + any pending
  `instructions`/`thinking` context appended since the last turn.
- Calls Ollama `/api/chat` (or `/api/generate`) with `stream: true`.
- Buffers tokens until a sentence boundary (`. ! ? ` or newline), then:
  - sends that sentence to Kokoro for synthesis,
  - pushes the resulting PCM into `OutputAudioTrack`'s outgoing queue,
  - emits `session.output_transcript.delta` for that sentence.
- Repeats until the model finishes or the task is cancelled (barge-in).
- Delegation trigger: if the model's text contains a recognizable marker
  (e.g. a line matching `[[SEARCH: <query>]]`), strip that marker from the
  spoken output, emit `session.delegation.created`
  `{delegation:{target:"client", id: <uuid>}}` for that query, and treat
  the rest of the sentence normally. The Android app already knows how to
  handle this (`MuralViewModel.delegate(id)` → calls `respond()` with
  `search:true` → sends `session.commentary.append` back with the id).
  The prompt/instructions sent to Ollama must document this marker so the
  model knows to use it.

### Barge-in
- While `ResponsePipeline` has audio queued or is generating, `InputPipeline`
  keeps evaluating VAD on the incoming track.
- Sustained speech (>200ms above the VAD threshold) while the bot is
  speaking cancels the `ResponsePipeline` asyncio task, drains/clears
  `OutputAudioTrack`'s pending queue, and starts a new turn accumulation
  immediately — mirrors the interruption UX the app already assumes
  (mute/unmute events aside, the existing UI has no separate "assistant was
  interrupted" event, so no new event type is needed here).
- Implemented as: one `asyncio.Event`/task handle per session; input path
  sets a "barge-in requested" flag the response task checks between
  sentence-synthesis steps, plus outright `task.cancel()` for the common
  case.

### OutputAudioTrack
- Custom `aiortc.mediastreams.MediaStreamTrack` (kind `"audio"`) backed by
  an `asyncio.Queue` of PCM frames. `ResponsePipeline` pushes frames as
  Kokoro produces them; `recv()` pops and paces them at the WebRTC clock
  rate. Barge-in cancellation clears the queue.

### /responses endpoint (delegation + standalone lesson/quiz generation)
- Accepts the same body shape Android already sends (`APIClient.kt`
  `respond()`: `model`, `instructions`, `input`, `max_output_tokens`,
  `reasoning.effort`, optional `text.format.json_schema` with
  `strict:true`, optional `tools:[{"type":"web_search"}]` +
  `tool_choice:"auto"` + `max_tool_calls:1`).
- Translates to Ollama:
  - `format` = the JSON schema object as-is when `text.format` is present
    (Ollama ≥0.5 structured outputs take a JSON-schema object directly).
  - When `tools` includes `web_search`, register a single Ollama tool
    `web_search(query: string)` and run Ollama's tool-calling loop capped
    at 1 call (matches `max_tool_calls:1`): if the model calls it, execute
    via `duckduckgo-search`, feed the top ~3 results' titles+snippets+URLs
    back as the tool result, then get the final structured response.
- Returns a response shaped like OpenAI's `/responses` output closely
  enough for `decodeTeachingResponse` in `APIClient.kt` to parse (need to
  check that decoder's exact expected fields when implementing — flagged
  as an implementation-time detail, not a design gap, since the shape is
  fully under our control on both ends... except `APIClient.kt` isn't
  being changed, so the server must match its existing parser exactly).

### Usage/session lifecycle
- `usage.seconds` = wall-clock seconds since session start, sent on every
  `session.usage.updated` (periodic, e.g. every 5s, cheap) and in the final
  `session.closed`. No billing semantics — purely informational, matches
  what `MuralViewModel.handle()` already does with the number (display
  only in local mode).
- `session.close` (from client) or data-channel/peer-connection teardown →
  server sends `session.closed`, then tears down the `Session`.

## Android client changes

Minimal by design, because the wire protocol doesn't change:

1. **`ConversationProviders.kt`**: add `LOCAL_SERVER` to
   `enum class ConversationProvider { PERSONAL_KEY, HOSTED_MINUTES }`.
2. **`APIClient.kt`**:
   - `baseUrl` is already a constructor parameter defaulting to
     `API_BASE_URL` (`api.openai.com`) — expose a way to construct it
     pointed at `http://<local-ip>:8000/` for local mode.
   - `post()` currently throws `APIException.MissingKey` when
     `readCredential()` returns null; add a `requiresAuth` flag (or just
     have local mode's `readCredential` return a dummy non-null value and
     skip attaching the `Authorization` header when `baseUrl` is not
     `api.openai.com`) so no OpenAI key is required in local mode.
3. **`SettingsScreen.kt`**: add a "Local server" option alongside the
   existing key-entry UI — a text field for `host:port`, stored via the
   existing preferences/credential storage mechanism (reuse, don't build a
   new store).
4. **`MuralViewModel.kt`**: in the provider-selection path (around line
   653 where `LiveSessionProvider` is chosen per provider), add
   `LOCAL_SERVER -> api` (same `APIClient` instance, just constructed with
   the local `baseUrl`). No changes to `handle(event)` — the existing
   event switch already covers 100% of the vocabulary the local server
   emits.
5. **`cloudReady()`** gating (line ~311, currently checks
   `PERSONAL_KEY && !hasKey`) needs a matching local-mode check (server
   address configured) instead of an OpenAI key check.

No changes to `LiveTransport.kt`, `LiveTransport`'s WebRTC setup, audio
routing, metering, or barge-in-adjacent UI — all of that is provider-agnostic
already.

## Error handling

- Server unreachable / connection refused → surfaces through the same
  `TransportException.Connection` / `Timeout` paths already in
  `LiveTransport.kt` (`createLiveSession` throwing, or ICE/SDP timeouts) —
  no new error plumbing needed on the client.
- Ollama not running / model not pulled → `/live/sessions` and
  `/responses` return HTTP 503 with a clear message; server logs which
  step failed (STT/LLM/TTS) for local debugging.
- Whisper/Kokoro exceptions mid-turn → caught per-turn, session emits an
  `error` event (already handled by `MuralViewModel` → shows
  `notice_voice_update_rejected`) rather than tearing down the whole
  connection.
- DuckDuckGo search failure/timeout → treated as "no results", model
  proceeds without search context rather than failing the whole delegation.

## Testing strategy

- **Unit**: VAD turn-boundary detection against fixture WAV clips
  (silence/speech/silence); barge-in cancellation logic with a fake
  `ResponsePipeline`; Ollama structured-output translation (valid/invalid
  schema round-trips); DuckDuckGo tool wrapper (mocked HTTP).
- **Integration**: a small `aiortc`-based Python test client that performs
  the full SDP offer/answer + sends synthetic audio + asserts on the
  received event sequence (`mural.session.created` → `session.started` →
  `session.input_transcript.delta` → `session.output_transcript.delta` →
  `session.closed`), run against the real server process.
- **Manual**: physical Android device on the same Wi-Fi as the Mac running
  the server; full conversation including talking over the bot mid-reply
  to confirm barge-in actually cancels and restarts a turn; a lesson/quiz
  generation call to confirm schema compliance end-to-end.

## Open implementation-time details (not design gaps, just deferred)

- Exact Ollama tool-calling request/response shape for `qwen2.5:7b` should
  be verified against the installed Ollama version before writing
  `DelegationBridge`/`/responses` — API surface has changed across Ollama
  releases.
- `decodeTeachingResponse` in `APIClient.kt` (not shown in this spec) must
  be read during implementation to confirm the exact JSON field names the
  local `/responses` endpoint needs to return.
