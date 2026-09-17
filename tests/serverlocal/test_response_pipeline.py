import asyncio

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
