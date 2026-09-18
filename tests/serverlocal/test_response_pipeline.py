import asyncio
import time

import numpy as np

from serverlocal.response_pipeline import ResponsePipeline, chunk_sentences


def test_chunk_sentences_splits_on_terminators():
    tokens = ["Hello", " world.", " How are", " you?", " Extra"]
    assert list(chunk_sentences(tokens)) == ["Hello world.", "How are you?", "Extra"]


def test_chunk_sentences_handles_no_terminator_at_all():
    assert list(chunk_sentences(["just", " text", " no end"])) == ["just text no end"]


def test_chunk_sentences_handles_empty_stream():
    assert list(chunk_sentences([])) == []


def test_response_pipeline_streams_sentences_to_audio_and_deltas():
    deltas = []
    pushed = []

    def stream_reply_fn(prompt):
        assert prompt == "hola"
        return iter(["Hello", " world.", " Bye."])

    def synthesize_fn(text):
        return np.zeros(1600, dtype=np.int16), 16000  # 100ms @ 16kHz

    async def push_audio_fn(samples, sample_rate):
        pushed.append((len(samples), sample_rate))

    def on_output_delta(text, start_ms, end_ms):
        deltas.append((text, start_ms, end_ms))

    pipeline = ResponsePipeline(stream_reply_fn, synthesize_fn, push_audio_fn, on_output_delta)
    full_text = asyncio.run(pipeline.run("hola"))

    assert full_text == "Hello world. Bye."
    assert deltas == [("Hello world.", 0, 100), ("Bye.", 100, 200)]
    assert pushed == [(1600, 16000), (1600, 16000)]


def test_run_lets_the_event_loop_keep_running_during_blocking_generation_and_synthesis():
    """Real Ollama/Kokoro calls block synchronously for real wall-clock
    time. If run() doesn't get that work off the event loop, a
    concurrently-running task (in production: _consume_audio, which is
    what makes barge-in actually able to fire) gets starved for the
    whole turn instead of just between sentences."""

    def slow_stream_reply_fn(prompt):
        time.sleep(0.05)
        yield "One sentence. "
        time.sleep(0.05)
        yield "Another sentence."

    def slow_synthesize_fn(text):
        time.sleep(0.05)
        return np.zeros(1600, dtype=np.int16), 16000

    async def push_audio_fn(samples, sample_rate):
        pass

    def on_output_delta(text, start_ms, end_ms):
        pass

    pipeline = ResponsePipeline(slow_stream_reply_fn, slow_synthesize_fn, push_audio_fn, on_output_delta)

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        ticker_task = asyncio.ensure_future(ticker())
        await pipeline.run("hola")
        ticker_task.cancel()
        return ticks

    ticks = asyncio.run(scenario())
    # ~200ms of blocking work spread across real awaits, ticking every
    # 10ms, should let the ticker run many times -- a blocked loop would
    # leave this at 0 or 1.
    assert ticks > 5
