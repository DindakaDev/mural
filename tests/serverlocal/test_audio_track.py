import asyncio

import numpy as np

from serverlocal.audio_track import OutputAudioTrack


def test_recv_returns_frame_with_correct_rate_and_increasing_pts():
    track = OutputAudioTrack()
    samples = np.array([100, -100, 200, -200], dtype=np.int16)

    async def scenario():
        await track.push_pcm(samples, sample_rate=24000)
        frame1 = await track.recv()
        await track.push_pcm(samples, sample_rate=24000)
        frame2 = await track.recv()
        return frame1, frame2

    frame1, frame2 = asyncio.run(scenario())
    assert frame1.sample_rate == 24000
    assert frame1.samples == 4
    assert frame1.pts == 0
    assert frame2.pts == 4


def test_clear_drops_queued_but_unsent_frames():
    track = OutputAudioTrack()
    samples = np.zeros(10, dtype=np.int16)

    async def scenario():
        await track.push_pcm(samples, sample_rate=16000)
        await track.push_pcm(samples, sample_rate=16000)
        track.clear()
        return track._queue.qsize()

    assert asyncio.run(scenario()) == 0
