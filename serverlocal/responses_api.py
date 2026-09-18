from fastapi import APIRouter
from pydantic import BaseModel

from .ollama_responses import generate

router = APIRouter()


class ResponsesBody(BaseModel):
    model: str = "qwen2.5:7b"
    instructions: str = ""
    input: list[dict]
    store: bool | None = None
    max_output_tokens: int | None = None
    reasoning: dict | None = None
    text: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | None = None
    max_tool_calls: int | None = None


def _extract_user_text(input_items: list[dict]) -> str:
    return "\n".join(item["content"] for item in input_items if isinstance(item.get("content"), str))


def _extract_schema(body: ResponsesBody) -> dict | None:
    if not body.text:
        return None
    fmt = body.text.get("format") or {}
    if fmt.get("type") != "json_schema":
        return None
    return fmt.get("schema")


def _wants_search(body: ResponsesBody) -> bool:
    return bool(body.tools) and any(t.get("type") == "web_search" for t in body.tools)


@router.post("/responses")
async def responses_route(body: ResponsesBody) -> dict:
    user_text = _extract_user_text(body.input)
    schema = _extract_schema(body)
    want_search = _wants_search(body)

    result = generate(body.instructions, user_text, schema, want_search, body.model)

    output: list[dict] = []
    if result.searches:
        output.append({"type": "web_search_call"})
    output.append({
        "content": [{
            "type": "output_text",
            "text": result.text,
            "annotations": [
                {"type": "url_citation", "url": source["url"], "title": source.get("title", "")}
                for source in result.sources
            ],
        }],
    })

    return {
        "status": "completed",
        "output": output,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
    }
