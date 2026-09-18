import re
from collections.abc import Callable, Iterable, Iterator

SEARCH_MARKER = re.compile(r"\[\[SEARCH:\s*(.*?)\]\]")


def strip_delegation_markers(token_stream: Iterable[str], on_delegation: Callable[[str], None]) -> Iterator[str]:
    """Wraps a streamed-token generator, detecting [[SEARCH: query]]
    markers that may span multiple tokens. Calls on_delegation(query) for
    each complete marker found and yields the token stream with markers
    removed. Text that looks like it MIGHT be the start of a marker is
    withheld until either the marker completes (and is stripped) or the
    stream proves it wasn't one (flushed as plain text, including at
    stream end)."""
    buffer = ""
    for token in token_stream:
        buffer += token
        while True:
            match = SEARCH_MARKER.search(buffer)
            if not match:
                break
            on_delegation(match.group(1).strip())
            buffer = buffer[: match.start()] + buffer[match.end() :]
        safe_end = _safe_flush_point(buffer)
        if safe_end:
            yield buffer[:safe_end]
            buffer = buffer[safe_end:]
    if buffer:
        yield buffer


def _safe_flush_point(buffer: str) -> int:
    """How much of buffer can be safely yielded without risking splitting
    a still-forming marker. Withholds everything from the last "[[" if it
    hasn't yet resolved into a confirmed non-marker or a completed one."""
    idx = buffer.rfind("[[")
    if idx == -1:
        return len(buffer)
    return idx
