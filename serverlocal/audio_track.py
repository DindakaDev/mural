import asyncio
import fractions
import time

import av
import numpy as np
from aiortc.mediastreams import MediaStreamTrack

FRAME_MS = 20


class OutputAudioTrack(MediaStreamTrack):
    """Outgoing WebRTC audio track fed by pushing PCM chunks as they're
    synthesized. aiortc's Opus encoder resamples whatever format/rate it's
    given to 48kHz stereo internally, so this track passes through
    Kokoro's native mono output untouched.

    push_pcm() splits whatever it's given into 20ms sub-frames so that
    each RTP packet aiortc's encoder produces gets its own frame/pts
    (aiortc's Opus encoder splits any longer frame into multiple packets
    but only advances sequence_number between them, leaving them sharing
    one RTP timestamp). recv() paces itself against a wall clock so the
    track's timeline doesn't drift from real time when a caller pushes a
    burst of audio all at once (e.g. a whole synthesized sentence)."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._timestamp = 0
        self._start: float | None = None

    async def push_pcm(self, samples: np.ndarray, sample_rate: int) -> None:
        if np.issubdtype(samples.dtype, np.floating):
            # Kokoro (and other float synthesizers) hand back float32/64
            # PCM in [-1.0, 1.0]. `.astype(np.int16)` on those truncates
            # toward zero instead of scaling, producing silence.
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            pcm16 = samples.astype(np.int16)

        chunk_size = max(1, round(sample_rate * FRAME_MS / 1000))
        for start in range(0, len(pcm16), chunk_size):
            chunk = pcm16[start : start + chunk_size]
            frame = av.AudioFrame.from_ndarray(
                chunk.reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate = sample_rate
            await self._queue.put(frame)

    def clear(self) -> None:
        """Drops queued-but-unsent audio. A no-op if nothing is queued."""
        while not self._queue.empty():
            self._queue.get_nowait()

    def has_queued_audio(self) -> bool:
        return not self._queue.empty()

    async def recv(self) -> av.AudioFrame:
        frame = await self._queue.get()
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, frame.sample_rate)
        self._timestamp += frame.samples

        now = time.monotonic()
        if self._start is None:
            # First frame goes out immediately; every later frame is
            # paced against this reference point so the whole track's
            # timeline tracks the wall clock instead of the queue's
            # arrival times.
            self._start = now
        else:
            # A gap with nothing queued (a turn boundary, a future
            # barge-in clear()) leaves _start behind wall clock. Without
            # re-anchoring, every frame pushed after the gap computes a
            # `wait` that's already in the past and gets released
            # instantly -- an unpaced burst instead of real-time audio.
            # max() only ever moves the anchor forward, so a track that's
            # still on pace is untouched.
            self._start = max(self._start, now - self._timestamp / frame.sample_rate)

        wait = self._start + (self._timestamp / frame.sample_rate) - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)

        return frame
