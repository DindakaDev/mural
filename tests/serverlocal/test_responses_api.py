from unittest.mock import MagicMock

from serverlocal import responses_api
from serverlocal.ollama_responses import ResponseResult


def test_responses_route_builds_decodeteachingresponse_compatible_shape(monkeypatch):
    fake_generate = MagicMock(return_value=ResponseResult(text="Hola", input_tokens=12, output_tokens=7))
    monkeypatch.setattr(responses_api, "generate", fake_generate)

    body = responses_api.ResponsesBody(
        model="qwen2.5:7b",
        instructions="Reply briefly.",
        input=[{"role": "user", "content": "hola"}],
    )

    import asyncio

    result = asyncio.run(responses_api.responses_route(body))

    assert result["status"] == "completed"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 7}
    content = result["output"][-1]["content"][0]
    assert content["type"] == "output_text"
    assert content["text"] == "Hola"
    assert content["annotations"] == []
    assert not any(item.get("type") == "web_search_call" for item in result["output"])
    fake_generate.assert_called_once_with("Reply briefly.", "hola", None, False, "qwen2.5:7b")


def test_responses_route_extracts_schema_and_search_flag(monkeypatch):
    fake_generate = MagicMock(return_value=ResponseResult(text="ok"))
    monkeypatch.setattr(responses_api, "generate", fake_generate)

    schema = {"type": "object", "properties": {}}
    body = responses_api.ResponsesBody(
        model="qwen2.5:7b",
        instructions="",
        input=[{"role": "user", "content": "q"}],
        text={"format": {"type": "json_schema", "name": "mural_result", "strict": True, "schema": schema}},
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        max_tool_calls=1,
    )

    import asyncio

    asyncio.run(responses_api.responses_route(body))

    fake_generate.assert_called_once_with("", "q", schema, True, "qwen2.5:7b")


def test_responses_route_includes_web_search_call_marker_and_citations_when_searched(monkeypatch):
    fake_generate = MagicMock(return_value=ResponseResult(
        text="Lima is the capital.",
        sources=[{"url": "https://example.com/peru", "title": "Peru"}],
        searches=1,
    ))
    monkeypatch.setattr(responses_api, "generate", fake_generate)

    body = responses_api.ResponsesBody(model="qwen2.5:7b", instructions="", input=[{"role": "user", "content": "q"}])

    import asyncio

    result = asyncio.run(responses_api.responses_route(body))

    assert result["output"][0] == {"type": "web_search_call"}
    annotations = result["output"][1]["content"][0]["annotations"]
    assert annotations == [{"type": "url_citation", "url": "https://example.com/peru", "title": "Peru"}]


def test_responses_route_joins_multiple_string_input_items(monkeypatch):
    fake_generate = MagicMock(return_value=ResponseResult(text="ok"))
    monkeypatch.setattr(responses_api, "generate", fake_generate)

    body = responses_api.ResponsesBody(
        model="qwen2.5:7b",
        instructions="",
        input=[{"role": "user", "content": "first"}, {"role": "user", "content": "second"}],
    )

    import asyncio

    asyncio.run(responses_api.responses_route(body))

    fake_generate.assert_called_once_with("", "first\nsecond", None, False, "qwen2.5:7b")
