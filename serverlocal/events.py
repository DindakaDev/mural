import uuid


def session_created(session_id: str) -> dict:
    return {"type": "mural.session.created", "session": {"id": session_id}}


def session_started() -> dict:
    return {"type": "session.started"}


def transcript_delta(kind: str, text: str, start_ms: int, end_ms: int) -> dict:
    if kind not in ("input", "output"):
        raise ValueError(f"kind must be 'input' or 'output', got {kind!r}")
    return {
        "type": f"session.{kind}_transcript.delta",
        "delta": text,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "event_id": str(uuid.uuid4()),
    }


def usage_updated(seconds: float, closed: bool = False) -> dict:
    return {
        "type": "session.closed" if closed else "session.usage.updated",
        "usage": {"seconds": seconds},
    }


def error(message: str) -> dict:
    return {"type": "error", "message": message}


def delegation_created(delegation_id: str) -> dict:
    return {"type": "session.delegation.created", "delegation": {"target": "client", "id": delegation_id}}
