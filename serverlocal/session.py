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

DEFAULT_SYSTEM_PROMPT = "Reply in the same language as the user input in 1 short sentence."


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
        self.instructions: str = ""
        self.language: str | None = None

    def send(self, event: dict) -> None:
        if self.channel is not None and self.channel.readyState == "open":
            self.channel.send(json.dumps(event))

    def usage_seconds(self) -> float:
        return time.monotonic() - self.started_at


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
        session.language = language
        session.send(events.transcript_delta("input", text, start_ms, end_ms))
        asyncio.ensure_future(run_response_turn(session, text))

    return InputPipeline(transcribe_fn, on_turn_transcribed)


async def run_response_turn(session: Session, user_text: str) -> None:
    system_prompt = session.instructions if session.instructions else DEFAULT_SYSTEM_PROMPT
    prompt = f"{system_prompt}\nUser: {user_text}\nAssistant:"

    def stream_reply_fn(p: str):
        return ollama_client.stream_reply(p)

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


async def create_live_session(body: LiveSessionBody) -> dict:
    offer_sdp = body.transport["sdp"]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    session = Session(pc)
    session.instructions = body.session.get("instructions", "") or ""
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
            asyncio.ensure_future(_consume_audio(track, input_pipeline, session))

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


@router.post("/live/sessions")
async def live_sessions_route(body: LiveSessionBody) -> dict:
    return await create_live_session(body)
