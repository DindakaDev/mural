from unittest.mock import MagicMock, patch

from serverlocal import search_tool


def test_web_search_returns_url_title_and_body(monkeypatch):
    fake_results = [
        {"href": "https://example.com/a", "title": "A", "body": "About A"},
        {"href": "https://example.com/b", "title": "B", "body": "About B"},
    ]
    fake_ddgs = MagicMock()
    fake_ddgs.__enter__.return_value.text.return_value = iter(fake_results)
    monkeypatch.setattr(search_tool, "DDGS", lambda: fake_ddgs)

    results = search_tool.web_search("test query", max_results=2)

    assert results == [
        {"url": "https://example.com/a", "title": "A", "body": "About A"},
        {"url": "https://example.com/b", "title": "B", "body": "About B"},
    ]
    fake_ddgs.__enter__.return_value.text.assert_called_once_with("test query", max_results=2)


def test_web_search_drops_results_with_no_url(monkeypatch):
    fake_results = [{"href": "", "title": "No URL", "body": "x"}]
    fake_ddgs = MagicMock()
    fake_ddgs.__enter__.return_value.text.return_value = iter(fake_results)
    monkeypatch.setattr(search_tool, "DDGS", lambda: fake_ddgs)

    assert search_tool.web_search("q") == []


def test_web_search_returns_empty_list_on_failure(monkeypatch):
    def raise_error():
        raise RuntimeError("network down")

    monkeypatch.setattr(search_tool, "DDGS", raise_error)

    assert search_tool.web_search("q") == []


def test_web_search_tool_definition_matches_ollama_function_tool_shape():
    assert search_tool.WEB_SEARCH_TOOL["type"] == "function"
    function = search_tool.WEB_SEARCH_TOOL["function"]
    assert function["name"] == "web_search"
    assert function["parameters"]["required"] == ["query"]
    assert "query" in function["parameters"]["properties"]
