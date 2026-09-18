from dataclasses import dataclass, field

import requests

from . import search_tool

OLLAMA_HOST = "http://localhost:11434"


@dataclass
class ResponseResult:
    text: str
    sources: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0


def _chat(messages: list[dict], model: str, format_schema: dict | None = None, tools: list[dict] | None = None) -> dict:
    payload: dict = {"model": model, "messages": messages, "stream": False}
    if format_schema is not None:
        payload["format"] = format_schema
    if tools is not None:
        payload["tools"] = tools
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _format_search_results(results: list[dict]) -> str:
    if not results:
        return "No search results found."
    lines = [f"- {r['title']}: {r['body']} ({r['url']})" for r in results]
    return "Search results:\n" + "\n".join(lines)


def generate(instructions: str, user_text: str, schema: dict | None, want_search: bool, model: str = "qwen2.5:7b") -> ResponseResult:
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": user_text})

    sources: list[dict] = []
    searches = 0

    if want_search:
        first = _chat(messages, model, tools=[search_tool.WEB_SEARCH_TOOL])
        tool_calls = first.get("message", {}).get("tool_calls") or []
        if tool_calls:
            query = tool_calls[0].get("function", {}).get("arguments", {}).get("query", user_text)
            results = search_tool.web_search(query)
            searches = 1
            sources = results
            messages.append(first["message"])
            messages.append({"role": "tool", "content": _format_search_results(results)})
            final = _chat(messages, model, format_schema=schema)
        elif schema is not None:
            final = _chat(messages, model, format_schema=schema)
        else:
            final = first
    else:
        final = _chat(messages, model, format_schema=schema)

    return ResponseResult(
        text=final.get("message", {}).get("content", ""),
        sources=sources,
        input_tokens=final.get("prompt_eval_count", 0),
        output_tokens=final.get("eval_count", 0),
        searches=searches,
    )
