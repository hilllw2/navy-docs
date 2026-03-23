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
    book_map = {b["source_file"]: b for b in catalog}

    api_key = get_gemini_api_key()
    text_model, _ = get_models()
    client = genai.Client(api_key=api_key)

    catalog_text = "\n".join(
        [
            "- source_file: {src} | title: {title} | focus: {summary} | aliases: {aliases}".format(
                src=b["source_file"],
                title=b.get("title", ""),
                summary=b.get("summary", ""),
                aliases=", ".join(b.get("aliases", [])) or "none",
            )
            for b in catalog
        ]
    )

    prompt = (
        "You are a naval knowledge router. Analyze the user's question and match it to the most relevant training manual.\n\n"
        "TASK: Return JSON with these keys:\n"
        "- refined_query: Clarify the question using naval terminology (keep it simple for straightforward questions)\n"
        "- target_source_file: Exact source_file from catalog or null\n"
        "- routing_confidence: 0.0-1.0 (use 0.8+ for clear matches, 0.5-0.7 for uncertain, <0.5 for no match)\n"
        "- route_reason_short: Brief explanation (max 15 words)\n\n"
        "GUIDELINES:\n"
        "- For simple, direct questions: keep refined_query close to original\n"
        "- Match book focus to question topic (e.g., towing→BR45 vol6, collision rules→COLREG)\n"
        "- If question is clear and book is obvious, set confidence 0.85+\n"
        "- Return JSON only, no markdown\n\n"
        f"AVAILABLE BOOKS:\n{catalog_text}\n\n"
        f"RECENT CONTEXT:\n{conversation_context or 'none'}\n\n"
        f"USER QUESTION:\n{query}\n"
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

    selected_book = None

    if not route["target_source_file"]:
        fallback = _alias_fallback(query, catalog)
        if fallback:
            route["target_source_file"] = fallback
            route["routing_confidence"] = max(route["routing_confidence"], 0.65)
            route["route_reason_short"] = "Alias-based fallback match."
            selected_book = book_map.get(fallback)
    else:
        selected_book = book_map.get(route["target_source_file"])

    valid_sources = {b["source_file"] for b in catalog}
    if source_file_lock in valid_sources:
        route["target_source_file"] = source_file_lock
        route["routing_confidence"] = 1.0
        route["route_reason_short"] = "User enabled source-file lock."
        selected_book = book_map.get(source_file_lock)

    if not selected_book and route["target_source_file"]:
        selected_book = book_map.get(route["target_source_file"])

    if selected_book:
        hint = (
            f"{selected_book.get('title', selected_book['source_file'])} — {selected_book.get('summary', '').strip()}"
        )
    else:
        hint = "Royal Navy navigation manuals covering COLREGS, shiphandling, radar, and emergency procedures."

    state["route"] = route
    state["route_debug"] = {
        "catalog_size": len(catalog),
        "confidence_threshold": 0.7,
    }
    state["book_context_hint"] = hint
    return state
