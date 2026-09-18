from duckduckgo_search import DDGS

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information needed to answer the user's question.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Runs a DuckDuckGo text search. Returns an empty list on any
    failure (network down, DuckDuckGo rate limit, etc) rather than
    raising -- a failed search should degrade the model's answer, not
    crash the request."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []
    return [
        {"url": r["href"], "title": r.get("title", ""), "body": r.get("body", "")}
        for r in results
        if r.get("href")
    ]
