import asyncio
import fractions
import json

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
    pipeline._on_turn("hola de prueba", "es", 0, 500)

    assert len(sent) == 1
    assert sent[0]["type"] == "session.input_transcript.delta"
    assert sent[0]["delta"] == "hola de prueba"
    assert session.language == "es"


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


def test_run_response_turn_uses_client_instructions_as_system_prompt(monkeypatch):
    captured = {}

    def fake_stream_reply(prompt):
        captured["prompt"] = prompt
        return iter(["Hi."])

    monkeypatch.setattr(ollama_client, "stream_reply", fake_stream_reply)
    monkeypatch.setattr(models, "synthesize", lambda text, voice, lang: (np.zeros(160, dtype=np.int16), 16000))

    session = Session(pc=None)
    session.send = lambda event: None  # type: ignore[assignment]
    session.instructions = "Some custom instructions"

    class FakeTrack:
        async def push_pcm(self, samples, sample_rate):
            pass

    session.output_track = FakeTrack()  # type: ignore[assignment]

    asyncio.run(run_response_turn(session, "hola"))

    assert "Some custom instructions" in captured["prompt"]
    assert "hola" in captured["prompt"]


def test_run_response_turn_falls_back_to_default_system_prompt(monkeypatch):
    captured = {}

    def fake_stream_reply(prompt):
        captured["prompt"] = prompt
        return iter(["Hi."])

    monkeypatch.setattr(ollama_client, "stream_reply", fake_stream_reply)
    monkeypatch.setattr(models, "synthesize", lambda text, voice, lang: (np.zeros(160, dtype=np.int16), 16000))

    session = Session(pc=None)
    session.send = lambda event: None  # type: ignore[assignment]
    assert session.instructions == ""

    class FakeTrack:
        async def push_pcm(self, samples, sample_rate):
            pass

    session.output_track = FakeTrack()  # type: ignore[assignment]

    asyncio.run(run_response_turn(session, "hola"))

    assert "Reply in the same language as the user input in 1 short sentence." in captured["prompt"]


def test_create_live_session_stores_client_instructions_on_session(monkeypatch):
    captured_sessions = []
    original_init = Session.__init__

    def capturing_init(self, pc):
        original_init(self, pc)
        captured_sessions.append(self)

    monkeypatch.setattr(session_module.Session, "__init__", capturing_init)

    async def scenario():
        client_pc = RTCPeerConnection()
        client_pc.addTrack(_SilentMicTrack())
        client_pc.createDataChannel("oai-events")
        offer = await client_pc.createOffer()
        await client_pc.setLocalDescription(offer)

        body = LiveSessionBody(
            session={"instructions": "You are a helpful assistant."},
            transport={"type": "webrtc", "sdp": client_pc.localDescription.sdp},
        )
        await create_live_session(body)
        await client_pc.close()

    asyncio.run(scenario())

    assert len(captured_sessions) == 1
    assert captured_sessions[0].instructions == "You are a helpful assistant."


def test_run_response_turn_picks_english_voice_for_english_language(monkeypatch):
    captured = {}

    def fake_synthesize(text, voice, lang):
        captured["voice"] = voice
        captured["lang"] = lang
        return (np.zeros(160, dtype=np.int16), 16000)

    monkeypatch.setattr(models, "synthesize", fake_synthesize)
    monkeypatch.setattr(ollama_client, "stream_reply", lambda prompt: iter(["Hi."]))

    session = Session(pc=None)
    session.send = lambda event: None  # type: ignore[assignment]
    session.language = "en"

    class FakeTrack:
        async def push_pcm(self, samples, sample_rate):
            pass

    session.output_track = FakeTrack()  # type: ignore[assignment]

    asyncio.run(run_response_turn(session, "hello"))

    assert captured["voice"] == "af_bella"
    assert captured["lang"] == "en-us"


def test_run_response_turn_picks_spanish_voice_for_spanish_language(monkeypatch):
    captured = {}

    def fake_synthesize(text, voice, lang):
        captured["voice"] = voice
        captured["lang"] = lang
        return (np.zeros(160, dtype=np.int16), 16000)

    monkeypatch.setattr(models, "synthesize", fake_synthesize)
    monkeypatch.setattr(ollama_client, "stream_reply", lambda prompt: iter(["Hola."]))

    session = Session(pc=None)
    session.send = lambda event: None  # type: ignore[assignment]
    session.language = "es"

    class FakeTrack:
        async def push_pcm(self, samples, sample_rate):
            pass

    session.output_track = FakeTrack()  # type: ignore[assignment]

    asyncio.run(run_response_turn(session, "hola"))

    assert captured["voice"] == "ef_dora"
    assert captured["lang"] == "es"


def test_run_response_turn_defaults_to_english_voice_when_no_language_detected(monkeypatch):
    captured = {}

    def fake_synthesize(text, voice, lang):
        captured["voice"] = voice
        captured["lang"] = lang
        return (np.zeros(160, dtype=np.int16), 16000)

    monkeypatch.setattr(models, "synthesize", fake_synthesize)
    monkeypatch.setattr(ollama_client, "stream_reply", lambda prompt: iter(["Hi."]))

    session = Session(pc=None)
    session.send = lambda event: None  # type: ignore[assignment]
    assert session.language is None

    class FakeTrack:
        async def push_pcm(self, samples, sample_rate):
            pass

    session.output_track = FakeTrack()  # type: ignore[assignment]

    asyncio.run(run_response_turn(session, "hello"))

    assert captured["voice"] == "af_bella"
    assert captured["lang"] == "en-us"


def test_consume_audio_sends_error_event_and_keeps_listening_when_transcription_raises():
    sent = []
    session = Session(pc=None)
    session.send = sent.append  # type: ignore[assignment]

    class RaisingPipeline:
        def handle_frame(self, frame):
            raise RuntimeError("whisper exploded")

    class FakeTrack:
        def __init__(self):
            self._frames = [object(), object()]

        async def recv(self):
            if not self._frames:
                raise ConnectionError("track closed")
            return self._frames.pop(0)

    asyncio.run(session_module._consume_audio(FakeTrack(), RaisingPipeline(), session))

    # Both frames' failures were reported, and the loop kept running
    # (rather than dying on the first exception) until the track itself
    # signaled it was done.
    assert len(sent) == 2
    assert all(event["type"] == "error" for event in sent)
    assert sent[0]["message"] == "whisper exploded"
    assert sent[1]["message"] == "whisper exploded"


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


def test_pushed_audio_is_chunked_paced_and_non_silent_end_to_end():
    """Integration test proving C1 (float->int16 scaling) and C2 (20ms
    chunking + wall-clock pacing) actually work over a real two-peer
    aiortc connection with a real Opus encode/decode round-trip -- not
    just in isolated unit tests."""

    async def scenario():
        captured_sessions = []
        original_init = Session.__init__

        def capturing_init(self, pc):
            original_init(self, pc)
            captured_sessions.append(self)

        import unittest.mock as mock

        with mock.patch.object(Session, "__init__", capturing_init):
            client_pc = RTCPeerConnection()
            client_pc.addTrack(_SilentMicTrack())
            channel = client_pc.createDataChannel("oai-events")
            opened = asyncio.Event()

            @channel.on("open")
            def _on_open():
                opened.set()

            received_frames = []
            track_seen = asyncio.Event()

            @client_pc.on("track")
            def on_track(track):
                async def collect():
                    try:
                        while len(received_frames) < 8:
                            frame = await asyncio.wait_for(track.recv(), timeout=3)
                            received_frames.append(frame)
                    except Exception:
                        pass

                asyncio.ensure_future(collect())
                track_seen.set()

            offer = await client_pc.createOffer()
            await client_pc.setLocalDescription(offer)

            body = LiveSessionBody(session={}, transport={"type": "webrtc", "sdp": client_pc.localDescription.sdp})
            result = await create_live_session(body)
            await client_pc.setRemoteDescription(RTCSessionDescription(sdp=result["transport"]["sdp"], type="answer"))

            await asyncio.wait_for(opened.wait(), timeout=5)
            await asyncio.wait_for(track_seen.wait(), timeout=5)

            assert len(captured_sessions) == 1
            server_session = captured_sessions[0]

            sample_rate = 24000
            duration_s = 1.0
            t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

            await server_session.output_track.push_pcm(audio, sample_rate)

            for _ in range(200):
                if len(received_frames) >= 8:
                    break
                await asyncio.sleep(0.05)

            await client_pc.close()
            return received_frames

    frames = asyncio.run(scenario())

    # (a) more than one frame proves C2's chunking (a single unpaced
    # frame would arrive as one giant blob instead of many small ones).
    assert len(frames) > 1

    # (b) at least one frame decodes to non-silence, proving C1's
    # float->int16 scaling survived a real Opus encode/decode round trip
    # (exact sample values won't survive lossy Opus, but audible signal
    # will not come back as all zeros the way truncated silence would).
    non_zero_found = any(np.any(frame.to_ndarray() != 0) for frame in frames)
    assert non_zero_found

    # (c) consecutive frames have distinct, increasing timestamps.
    pts_values = [frame.pts for frame in frames]
    assert len(set(pts_values)) == len(pts_values)
    assert pts_values == sorted(pts_values)


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
