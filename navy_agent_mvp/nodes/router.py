from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from navy_agent_mvp.config import get_gemini_api_key, get_models, load_book_catalog
from navy_agent_mvp.state import AgentState
from navy_agent_mvp.utils import parse_json_loose


def _alias_fallback(query: str, catalog: List[Dict[str, Any]]) -> Optional[str]:
    q = query.lower()
    for book in catalog:
        source_file = book["source_file"]
        aliases = [source_file.lower(), book.get("title", "").lower(), *[a.lower() for a in book.get("aliases", [])]]
        if any(alias and alias in q for alias in aliases):
            return source_file
    return None


def route_query_node(state: AgentState) -> AgentState:
    query = state["user_query"]
    conversation_context = (state.get("conversation_context") or "").strip()
    source_file_lock = state.get("source_file_lock")
    catalog = load_book_catalog()

    api_key = get_gemini_api_key()
    text_model, _ = get_models()
    client = genai.Client(api_key=api_key)

    catalog_text = "\n".join(
        [
            f"- source_file: {b['source_file']} | title: {b.get('title', '')} | summary: {b.get('summary', '')} | aliases: {', '.join(b.get('aliases', []))}"
            for b in catalog
        ]
    )

    prompt = (
        "You are a naval knowledge router.\n"
        "Given the user question and a fixed book catalog, output STRICT JSON with keys:\n"
        "refined_query (string), target_source_file (string or null), routing_confidence (0..1 number), route_reason_short (string).\n"
        "Rules:\n"
        "1) target_source_file MUST be one exact source_file from catalog or null.\n"
        "2) refined_query should improve retrieval intent and keep original meaning.\n"
        "3) Keep route_reason_short under 25 words.\n"
        "4) Return JSON only.\n\n"
        f"CATALOG:\n{catalog_text}\n\n"
        f"SHORT_CONTEXT (recent chat summary):\n{conversation_context or 'none'}\n\n"
        f"USER_QUERY:\n{query}\n"
    )

    route = {
        "refined_query": query,
        "target_source_file": None,
        "routing_confidence": 0.0,
        "route_reason_short": "No routing decision.",
    }

    try:
        resp = client.models.generate_content(
            model=text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        data = parse_json_loose(resp.text or "")
        if isinstance(data, dict):
            target = data.get("target_source_file")
            valid_sources = {b["source_file"] for b in catalog}
            if target not in valid_sources:
                target = None

            confidence_raw = data.get("routing_confidence", 0.0)
            try:
                confidence = max(0.0, min(1.0, float(confidence_raw)))
            except (TypeError, ValueError):
                confidence = 0.0

            refined_query = data.get("refined_query")
            if not isinstance(refined_query, str) or not refined_query.strip():
                refined_query = query

            reason = data.get("route_reason_short")
            if not isinstance(reason, str) or not reason.strip():
                reason = "Model routing applied."

            route = {
                "refined_query": refined_query.strip(),
                "target_source_file": target,
                "routing_confidence": confidence,
                "route_reason_short": reason.strip()[:140],
            }
    except Exception:
        pass

    if not route["target_source_file"]:
        fallback = _alias_fallback(query, catalog)
        if fallback:
            route["target_source_file"] = fallback
            route["routing_confidence"] = max(route["routing_confidence"], 0.65)
            route["route_reason_short"] = "Alias-based fallback match."

    valid_sources = {b["source_file"] for b in catalog}
    if source_file_lock in valid_sources:
        route["target_source_file"] = source_file_lock
        route["routing_confidence"] = 1.0
        route["route_reason_short"] = "User enabled source-file lock."

    state["route"] = route
    state["route_debug"] = {
        "catalog_size": len(catalog),
        "confidence_threshold": 0.7,
    }
    return state
