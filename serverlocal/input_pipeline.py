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
