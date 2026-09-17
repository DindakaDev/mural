import asyncio
import fractions

import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack

from serverlocal import events, models, ollama_client, session as session_module
from serverlocal.session import LiveSessionBody, Session, build_input_pipeline, create_live_session, run_response_turn


class _SilentMicTrack(MediaStreamTrack):
    """Minimal fake microphone track so test offers include an audio
    m-line, matching real Android clients (LiveTransport.kt adds its
    local track before calling createOffer -- see LiveTransport.kt:274)."""

    kind = "audio"

    async def recv(self):
        samples = np.zeros((1, 320), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = 16000
        frame.pts = 0
        frame.time_base = fractions.Fraction(1, 16000)
        await asyncio.sleep(0.02)
        return frame


def test_create_live_session_returns_a_valid_answer():
    async def scenario():
        client_pc = RTCPeerConnection()
        client_pc.addTrack(_SilentMicTrack())
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
        client_pc.addTrack(_SilentMicTrack())
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


def test_run_response_turn_sends_error_event_when_ollama_raises(monkeypatch):
    def _raise(prompt):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr(ollama_client, "stream_reply", _raise)

    sent = []
    session = Session(pc=None)
    session.send = sent.append  # type: ignore[assignment]

    class FakeTrack:
        async def push_pcm(self, samples, sample_rate):
            raise AssertionError("push_pcm should not be called when stream_reply raises")

    session.output_track = FakeTrack()  # type: ignore[assignment]

    # Should not raise -- the exception must be caught and reported via
    # an error event instead of propagating out of the turn.
    asyncio.run(run_response_turn(session, "hola"))

    assert len(sent) == 1
    assert sent[0] == events.error("ollama unreachable")
