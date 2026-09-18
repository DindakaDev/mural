import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator

import numpy as np

SENTENCE_END = re.compile(r"[.!?\n]")


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
        barge-in (added in a later plan)."""
        elapsed_ms = 0
        parts: list[str] = []
        for sentence in chunk_sentences(self._stream_reply(prompt)):
            await asyncio.sleep(0)
            parts.append(sentence)
            samples, sample_rate = self._synthesize(sentence)
            await self._push_audio(samples, sample_rate)
            duration_ms = int(len(samples) / sample_rate * 1000)
            self._on_delta(sentence, elapsed_ms, elapsed_ms + duration_ms)
            elapsed_ms += duration_ms
        return " ".join(parts)
