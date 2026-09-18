import asyncio
import json
import time
import uuid

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from fastapi import APIRouter
from pydantic import BaseModel

from . import events, models, ollama_client
from .audio_track import OutputAudioTrack
from .delegation import strip_delegation_markers
from .input_pipeline import InputPipeline
from .response_pipeline import ResponsePipeline

router = APIRouter()

DEFAULT_SYSTEM_PROMPT = "Reply in the same language as the user input in 1 short sentence."
BARGE_IN_THRESHOLD_MS = 200
USAGE_BROADCAST_INTERVAL_SECONDS = 5
SEARCH_MARKER_INSTRUCTION = (
    "\nIf you need current information you don't already know, reply with "
    "exactly [[SEARCH: <query>]] and nothing else -- no other text."
)

_active_sessions: dict[str, "Session"] = {}


class LiveSessionBody(BaseModel):
    session: dict
    transport: dict


def _log(session_id: str, direction: str, event: dict) -> None:
    short_id = session_id[:8]
    kind = event.get("type", "?")
    detail = ""
    if "delta" in event:
        detail = f" {event['delta']!r}"[:80]
    elif "content" in event:
        detail = f" {event['content']!r}"[:80]
    elif "delegation" in event:
        detail = f" {event['delegation']}"
    elif "usage" in event:
        detail = f" {event['usage']}"
    elif "message" in event:
        detail = f" {event['message']!r}"[:80]
    print(f"[{short_id}] {direction} {kind}{detail}")


class Session:
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
        self.pending_context: list[str] = []
        self.muted: bool = False
        self.usage_task: asyncio.Task | None = None

    def send(self, event: dict) -> None:
        _log(self.id, "->", event)
        if self.channel is not None and self.channel.readyState == "open":
            self.channel.send(json.dumps(event))

    def usage_seconds(self) -> float:
        return time.monotonic() - self.started_at


def _cancel_current_response(session: Session) -> None:
    if session.response_task is not None and not session.response_task.done():
        session.response_task.cancel()
    session.output_track.clear()


def _voice_for_language(language: str | None) -> tuple[str, str]:
    """Mirrors the old server.py's female-voice mapping: Spanish gets
    ef_dora/es, everything else (including unknown/undetected language)
    falls back to English's af_bella/en-us."""
    if language and language.startswith("es"):
        return "ef_dora", "es"
    return "af_bella", "en-us"


def build_input_pipeline(session: Session) -> InputPipeline:
    def transcribe_fn(audio_np):
        return models.transcribe(audio_np)

    def on_turn_transcribed(text: str, language: str, start_ms: int, end_ms: int) -> None:
        if session.muted:
            return
        session.language = language
        session.send(events.transcript_delta("input", text, start_ms, end_ms))
        _cancel_current_response(session)
        session.response_task = asyncio.ensure_future(run_response_turn(session, text))

    return InputPipeline(transcribe_fn, on_turn_transcribed)


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


def _handle_client_message(session: Session, message) -> None:
    if not isinstance(message, str):
        return
    try:
        event = json.loads(message)
    except (ValueError, TypeError):
        return
    if not isinstance(event, dict):
        return
    _log(session.id, "<-", event)
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
    # Yield once so the cancellations scheduled above actually land (a
    # Task's cancelled() doesn't flip to True until the event loop gets
    # a chance to deliver CancelledError into it) before this function
    # returns and callers check response_task/usage_task.cancelled().
    # Same asyncio timing issue as _consume_audio's barge-in branch.
    await asyncio.sleep(0)
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


async def create_live_session(body: LiveSessionBody) -> dict:
    offer_sdp = body.transport["sdp"]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    session = Session(pc)
    session.instructions = body.session.get("instructions", "") or ""
    _active_sessions[session.id] = session
    print(f"[{session.id[:8]}] new session, instructions={session.instructions!r}")
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


async def _consume_audio(track, input_pipeline: InputPipeline, session: Session) -> None:
    while True:
        try:
            frame = await track.recv()
        except Exception:
            break
        try:
            input_pipeline.handle_frame(frame)
        except Exception as exc:
            # A single bad frame's transcription failure (e.g. Whisper
            # raising) must not kill audio consumption for the rest of
            # the session -- report it and keep listening.
            session.send(events.error(str(exc)))
            continue
        response_active = (
            session.response_task is not None and not session.response_task.done()
        ) or session.output_track.has_queued_audio()
        if (
            response_active
            and not session.muted
            and input_pipeline.current_speech_run_ms() >= BARGE_IN_THRESHOLD_MS
        ):
            _cancel_current_response(session)
            # Yield once so the cancellation actually lands on the
            # response task (task.cancel() only schedules delivery of
            # CancelledError; it doesn't take effect synchronously)
            # before this loop goes on to read the next audio frame.
            await asyncio.sleep(0)


@router.post("/live/sessions")
async def live_sessions_route(body: LiveSessionBody) -> dict:
    return await create_live_session(body)
