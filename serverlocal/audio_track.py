import asyncio
import fractions

import av
import numpy as np
from aiortc.mediastreams import MediaStreamTrack


class OutputAudioTrack(MediaStreamTrack):
    """Outgoing WebRTC audio track fed by pushing PCM chunks as they're
    synthesized. aiortc's Opus encoder resamples whatever format/rate it's
    given to 48kHz stereo internally, so this track passes through
    Kokoro's native mono output untouched."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._timestamp = 0

    async def push_pcm(self, samples: np.ndarray, sample_rate: int) -> None:
        frame = av.AudioFrame.from_ndarray(
            samples.astype(np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = sample_rate
        await self._queue.put(frame)

    def clear(self) -> None:
        """Drops queued-but-unsent audio. A no-op if nothing is queued."""
        while not self._queue.empty():
            self._queue.get_nowait()

    async def recv(self) -> av.AudioFrame:
        frame = await self._queue.get()
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, frame.sample_rate)
        self._timestamp += frame.samples
        return frame
