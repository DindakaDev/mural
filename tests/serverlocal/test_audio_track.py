import asyncio

import numpy as np

from serverlocal.audio_track import OutputAudioTrack


def test_recv_returns_frame_with_correct_rate_and_increasing_pts():
    # 4 samples is far less than one 20ms chunk at any realistic sample
    # rate, so each push_pcm call here still yields exactly one sub-frame.
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


def test_push_pcm_scales_float_samples_instead_of_truncating():
    # models.synthesize() (Kokoro) returns float32 audio in [-1, 1].
    # astype(np.int16) on those truncates toward zero and produces
    # silence -- push_pcm must scale instead.
    track = OutputAudioTrack()
    samples = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)

    async def scenario():
        await track.push_pcm(samples, sample_rate=24000)
        return await track.recv()

    frame = asyncio.run(scenario())
    result = frame.to_ndarray().reshape(-1)

    assert np.any(result != 0)
    assert result[0] == 16383  # 0.5 * 32767, truncated
    assert result[1] == -16383
    assert result[2] == 32767
    assert result[3] == -32767


def test_push_pcm_passes_through_int16_samples_unchanged():
    track = OutputAudioTrack()
    samples = np.array([100, -100, 200, -200], dtype=np.int16)

    async def scenario():
        await track.push_pcm(samples, sample_rate=24000)
        return await track.recv()

    frame = asyncio.run(scenario())
    result = frame.to_ndarray().reshape(-1)
    assert result.tolist() == [100, -100, 200, -200]


def test_push_pcm_chunks_long_audio_into_20ms_frames_with_pacing():
    # ~1 second of audio in one push_pcm call should come out of recv()
    # as ~50 20ms sub-frames (aiortc's Opus encoder otherwise splits one
    # long frame into multiple RTP packets that share a single
    # timestamp), each with a strictly increasing pts.
    track = OutputAudioTrack()
    sample_rate = 16000
    samples = np.full(sample_rate, 1000, dtype=np.int16)  # 1 second

    async def scenario():
        await track.push_pcm(samples, sample_rate=sample_rate)
        frames = []
        while not track._queue.empty():
            frames.append(await track.recv())
        return frames

    frames = asyncio.run(scenario())

    assert len(frames) == 50
    pts_values = [f.pts for f in frames]
    assert pts_values == sorted(pts_values)
    assert len(set(pts_values)) == len(pts_values)
    for f in frames[:-1]:
        assert f.samples == 320
    assert sum(f.samples for f in frames) == sample_rate


def test_recv_paces_frames_against_wall_clock():
    # recv() must not just return chunked frames instantly -- it should
    # release them roughly 20ms apart so the track's timeline tracks
    # real time instead of drifting.
    track = OutputAudioTrack()
    sample_rate = 16000
    samples = np.full(sample_rate // 5, 1000, dtype=np.int16)  # 200ms -> 10 frames

    async def scenario():
        import time

        await track.push_pcm(samples, sample_rate=sample_rate)
        start = time.time()
        for _ in range(10):
            await track.recv()
        return time.time() - start

    elapsed = asyncio.run(scenario())
    # 10 frames of 20ms each: first is immediate, so ~9 * 20ms = ~180ms
    # of real pacing should have elapsed. Generous lower bound to avoid
    # flakiness while still proving pacing (not instant return) happens.
    assert elapsed > 0.1
