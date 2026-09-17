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
