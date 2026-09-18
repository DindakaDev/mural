from serverlocal.delegation import strip_delegation_markers


def test_passes_through_text_with_no_marker():
    calls = []
    result = list(strip_delegation_markers(["Hello", " world."], calls.append))
    assert "".join(result) == "Hello world."
    assert calls == []


def test_extracts_a_marker_contained_in_one_token():
    calls = []
    result = list(strip_delegation_markers(["[[SEARCH: capital of Peru]]"], calls.append))
    assert "".join(result) == ""
    assert calls == ["capital of Peru"]


def test_extracts_a_marker_split_across_many_tokens():
    calls = []
    tokens = ["Sure, ", "[[SEARCH", ": weather in ", "Lima today", "]]", " done."]
    result = list(strip_delegation_markers(tokens, calls.append))
    assert "".join(result) == "Sure,  done."
    assert calls == ["weather in Lima today"]


def test_withholds_a_possible_partial_marker_until_the_stream_ends():
    calls = []
    # "[[SEA" alone could still become a marker -- must not be flushed
    # as plain text until the stream proves it isn't one.
    gen = strip_delegation_markers(iter(["before ", "[[SEA"]), calls.append)
    first = next(gen)
    assert first == "before "
    remainder = list(gen)
    # The stream ends without ever completing "]]" -- the partial text
    # is flushed as plain text rather than silently dropped.
    assert "".join(remainder) == "[[SEA"
    assert calls == []


def test_extracts_a_marker_containing_an_embedded_newline():
    calls = []
    result = list(strip_delegation_markers(["[[SEARCH: multi\nline query]]"], calls.append))
    assert "".join(result) == ""
    assert calls == ["multi\nline query"]


def test_extracts_a_marker_when_the_two_opening_brackets_are_split_across_tokens():
    calls = []
    tokens = ["Sure ", "[", "[SEARCH: capital of Peru]]", " ok"]
    result = list(strip_delegation_markers(iter(tokens), calls.append))
    assert "".join(result) == "Sure  ok"
    assert calls == ["capital of Peru"]
