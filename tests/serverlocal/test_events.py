from serverlocal import events


def test_session_created_shape():
    event = events.session_created("abc-123")
    assert event == {"type": "mural.session.created", "session": {"id": "abc-123"}}


def test_session_started_shape():
    assert events.session_started() == {"type": "session.started"}


def test_transcript_delta_input():
    event = events.transcript_delta("input", "hola", 100, 400)
    assert event["type"] == "session.input_transcript.delta"
    assert event["delta"] == "hola"
    assert event["start_ms"] == 100
    assert event["end_ms"] == 400
    assert isinstance(event["event_id"], str) and event["event_id"]


def test_transcript_delta_output():
    event = events.transcript_delta("output", "hi", 0, 200)
    assert event["type"] == "session.output_transcript.delta"


def test_transcript_delta_rejects_bad_kind():
    try:
        events.transcript_delta("bogus", "x", 0, 1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_usage_updated_shape():
    assert events.usage_updated(12.5) == {"type": "session.usage.updated", "usage": {"seconds": 12.5}}


def test_usage_updated_closed_shape():
    assert events.usage_updated(30.0, closed=True) == {"type": "session.closed", "usage": {"seconds": 30.0}}


def test_error_shape():
    assert events.error("bad thing") == {"type": "error", "message": "bad thing"}


def test_delegation_created_shape():
    event = events.delegation_created("abc-123")
    assert event == {
        "type": "session.delegation.created",
        "delegation": {"target": "client", "id": "abc-123"},
    }
