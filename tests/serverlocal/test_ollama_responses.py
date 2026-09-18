from unittest.mock import MagicMock

from serverlocal import ollama_responses


def _fake_response(content, prompt_eval_count=10, eval_count=5, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message, "prompt_eval_count": prompt_eval_count, "eval_count": eval_count}


def test_generate_without_search_makes_one_call_and_returns_its_content(monkeypatch):
    fake_post = MagicMock()
    fake_post.return_value.json.return_value = _fake_response("Hola mundo")
    fake_post.return_value.raise_for_status.return_value = None
    monkeypatch.setattr(ollama_responses.requests, "post", fake_post)

    result = ollama_responses.generate("Be brief.", "hola", schema=None, want_search=False, model="qwen2.5:7b")

    assert result.text == "Hola mundo"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.searches == 0
    assert result.sources == []
    assert fake_post.call_count == 1
    body = fake_post.call_args.kwargs["json"]
    assert body["model"] == "qwen2.5:7b"
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hola"}]
    assert "tools" not in body


def test_generate_with_schema_passes_it_as_format(monkeypatch):
    fake_post = MagicMock()
    fake_post.return_value.json.return_value = _fake_response('{"answer": "42"}')
    fake_post.return_value.raise_for_status.return_value = None
    monkeypatch.setattr(ollama_responses.requests, "post", fake_post)

    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = ollama_responses.generate("", "question", schema=schema, want_search=False, model="qwen2.5:7b")

    assert result.text == '{"answer": "42"}'
    body = fake_post.call_args.kwargs["json"]
    assert body["format"] == schema


def test_generate_with_search_calls_tool_once_then_returns_final_answer(monkeypatch):
    tool_call = {"function": {"name": "web_search", "arguments": {"query": "capital of Peru"}}}
    first = _fake_response("", tool_calls=[tool_call])
    final = _fake_response("The capital of Peru is Lima.", prompt_eval_count=20, eval_count=8)
    fake_post = MagicMock()
    fake_post.return_value.raise_for_status.return_value = None
    fake_post.return_value.json.side_effect = [first, final]
    monkeypatch.setattr(ollama_responses.requests, "post", fake_post)

    fake_search = MagicMock(return_value=[{"url": "https://example.com", "title": "Peru", "body": "Lima is..."}])
    monkeypatch.setattr(ollama_responses.search_tool, "web_search", fake_search)

    result = ollama_responses.generate("", "what is the capital of Peru?", schema=None, want_search=True, model="qwen2.5:7b")

    assert result.text == "The capital of Peru is Lima."
    assert result.searches == 1
    assert result.sources == [{"url": "https://example.com", "title": "Peru", "body": "Lima is..."}]
    assert result.input_tokens == 20
    assert result.output_tokens == 8
    fake_search.assert_called_once_with("capital of Peru")
    assert fake_post.call_count == 2
    first_body = fake_post.call_args_list[0].kwargs["json"]
    assert first_body["tools"] == [ollama_responses.search_tool.WEB_SEARCH_TOOL]
    second_body = fake_post.call_args_list[1].kwargs["json"]
    assert "tools" not in second_body
    assert second_body["messages"][-1]["role"] == "tool"


def test_generate_with_search_but_model_declines_to_call_it(monkeypatch):
    fake_post = MagicMock()
    fake_post.return_value.raise_for_status.return_value = None
    fake_post.return_value.json.return_value = _fake_response("I don't need to search for that.")
    monkeypatch.setattr(ollama_responses.requests, "post", fake_post)

    result = ollama_responses.generate("", "hello", schema=None, want_search=True, model="qwen2.5:7b")

    assert result.text == "I don't need to search for that."
    assert result.searches == 0
    assert fake_post.call_count == 1
