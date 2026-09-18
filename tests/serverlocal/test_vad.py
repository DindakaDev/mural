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
