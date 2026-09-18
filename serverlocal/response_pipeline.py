import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator

import numpy as np

SENTENCE_END = re.compile(r"[.!?\n]")
_NO_MORE_SENTENCES = object()


def chunk_sentences(token_stream: Iterable[str]) -> Iterator[str]:
    """Consumes an iterable of text chunks, yields complete sentences as
    they're formed. Any trailing text with no terminator is yielded once
    the stream ends."""
    buffer = ""
    for token in token_stream:
        buffer += token
        while True:
            match = SENTENCE_END.search(buffer)
            if not match:
                break
            end = match.end()
            sentence = buffer[:end].strip()
            buffer = buffer[end:]
            if sentence:
                yield sentence
    remainder = buffer.strip()
    if remainder:
        yield remainder


class ResponsePipeline:
    """Turns a user turn into streamed spoken output: LLM tokens -> sentence
    chunks -> TTS -> pushed onto the outgoing audio track, emitting one
    output_transcript.delta per sentence."""

    def __init__(
        self,
        stream_reply_fn: Callable[[str], Iterable[str]],
        synthesize_fn: Callable[[str], tuple[np.ndarray, int]],
        push_audio_fn: Callable[[np.ndarray, int], Awaitable[None]],
        on_output_delta: Callable[[str, int, int], None],
    ):
        self._stream_reply = stream_reply_fn
        self._synthesize = synthesize_fn
        self._push_audio = push_audio_fn
        self._on_delta = on_output_delta

    async def run(self, prompt: str) -> str:
        """Runs one full turn's reply and returns the full reply text.
        A caller may wrap this in an asyncio.Task and cancel it for
        barge-in.

        stream_reply_fn/synthesize_fn are blocking, synchronous calls
        (real HTTP streaming reads, real ONNX inference) -- both are run
        via asyncio.to_thread so the event loop stays free for the
        rest of the process (in particular, real-time audio consumption
        via _consume_audio, which is what lets barge-in actually notice
        the user talking) for the real duration of each sentence's
        generation and synthesis, not just at a single scheduling point
        between sentences.

        Cancelling the task this coroutine runs in only takes effect at
        one of these await points (matching the spec's "checks between
        sentence-synthesis steps" design) -- it cannot interrupt a
        single blocking to_thread call already in flight. That call's
        worker thread keeps running to completion in the background
        even after cancellation; its result is simply never awaited or
        used, since the caller already treats the turn as cancelled and
        clears any queued audio."""
        elapsed_ms = 0
        parts: list[str] = []
        sentences = chunk_sentences(self._stream_reply(prompt))
        while True:
            sentence = await asyncio.to_thread(next, sentences, _NO_MORE_SENTENCES)
            if sentence is _NO_MORE_SENTENCES:
                break
            parts.append(sentence)
            samples, sample_rate = await asyncio.to_thread(self._synthesize, sentence)
            await self._push_audio(samples, sample_rate)
            duration_ms = int(len(samples) / sample_rate * 1000)
            self._on_delta(sentence, elapsed_ms, elapsed_ms + duration_ms)
            elapsed_ms += duration_ms
        return " ".join(parts)
