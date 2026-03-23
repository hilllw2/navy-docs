from typing import List

from google import genai
from google.genai import types

from navy_agent_mvp.config import get_gemini_api_key, get_models
from navy_agent_mvp.state import AgentState
from navy_agent_mvp.utils import parse_json_loose

_TABLE_HINTS = {"table", "compare", "comparison", "vs", "versus", "difference", "differences", "matrix"}
_STEP_HINTS = {"steps", "step", "procedure", "process", "checklist", "how to", "how do", "what do"}
_CONCISE_HINTS = {"summary", "summarize", "brief", "short", "key", "most important", "top", "quick"}
_DETAILED_HINTS = {"explain", "detailed", "detail", "why", "rationale", "background"}


def _response_mode(question: str) -> dict:
    q = (question or "").lower()
    return {
        "table": any(h in q for h in _TABLE_HINTS),
        "steps": any(h in q for h in _STEP_HINTS),
        "concise": any(h in q for h in _CONCISE_HINTS) and not any(h in q for h in _DETAILED_HINTS),
        "detail": any(h in q for h in _DETAILED_HINTS),
    }


def _ensure_heading(text: str, heading_hint: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("### "):
        return text
    title = (heading_hint or "").strip().rstrip("?") or "Response"
    heading = f"### {title[:90]}"
    return f"{heading}\n\n{text}" if text.strip() else heading


def generate_topic_chat_response(user_query: str, short_memory: str, topic_context: str) -> str:
    api_key = get_gemini_api_key()
    text_model, _ = get_models()
    client = genai.Client(api_key=api_key)

    prompt = (
        "You are a helpful naval training assistant in free-chat mode.\n"
        "The user already searched the knowledge base. Use the provided topic context as anchor, "
        "but you may answer naturally like a normal AI assistant.\n\n"
        "RULES:\n"
        "- Begin every answer with a level-3 markdown heading that summarizes the takeaway.\n"
        "- Use short subheadings or bold labels for multi-part guidance.\n"
        "- Be direct, practical, and concise.\n"
        "- Prefer answers aligned with the provided topic context.\n"
        "- If the user asks outside the topic, still help and clearly say when uncertain.\n"
        "- Do not invent exact rule numbers or quoted passages unless present in context.\n"
        "- No JSON. Return plain markdown text only.\n\n"
        f"RECENT_MEMORY:\n{short_memory or 'none'}\n\n"
        f"TOPIC_CONTEXT:\n{topic_context or 'none'}\n\n"
        f"USER_QUESTION:\n{user_query}"
    )

    try:
        resp = client.models.generate_content(
            model=text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.35),
        )
        text = (resp.text or "").strip()
        if text:
            return _ensure_heading(text, user_query)
    except Exception:
        pass

    return _ensure_heading(
        "I can help with this topic, but I could not generate a response right now. Please try again.",
        user_query,
    )


def synthesize_answer_node(state: AgentState) -> AgentState:
    hits = state.get("hits", [])
    user_query = state["user_query"]
    refined_query = state["route"]["refined_query"]
    conversation_context = state.get("conversation_context") or ""
    plan = state.get("answer_plan") or {}
    book_hint = state.get("book_context_hint") or ""
    plan_heading = plan.get("heading") or user_query
    plan_sections = plan.get("sections") or []
    plan_style = plan.get("style_tips") or [
        "Be direct and confident.",
        "Use compact bullets for multiple steps.",
    ]
    
    # Check relevance of retrieved chunks - filter out low-quality matches
    RELEVANCE_THRESHOLD = 0.35  # Minimum similarity score to consider
    if hits:
        relevant_hits = [h for h in hits if h.get("similarity", 0.0) >= RELEVANCE_THRESHOLD]
        if len(relevant_hits) < len(hits) * 0.5:  # If more than half are below threshold
            # Evidence quality is poor, be more cautious
            hits = relevant_hits if relevant_hits else hits[:2]  # Keep at most 2 if all are poor

    section_lines: List[str] = []
    for idx, section in enumerate(plan_sections, start=1):
        title = (section.get("title") if isinstance(section, dict) else None) or f"Section {idx}"
        instruction = (section.get("instruction") if isinstance(section, dict) else None) or "Explain this part."
        section_lines.append(f"{idx}. {title}: {instruction}")
    if not section_lines:
        section_lines = ["1. Key Points: Summarize the most relevant facts."]

    style_line = "; ".join(plan_style)

    mode = _response_mode(user_query)

    if not hits:
        if conversation_context:
            style_hint = "Keep your answer brief and direct."
            if mode['detail']:
                style_hint = "Provide a clear explanation."
            elif mode['steps']:
                style_hint = "List the steps clearly."
            
            prompt = (
                "You are a Royal Navy training assistant. Answer based on the conversation context provided.\n\n"
                f"QUESTION: {user_query}\n\n"
                f"STYLE: {style_hint}\n\n"
                f"CONTEXT:\n{conversation_context}\n\n"
                "If the context doesn't contain the answer, say so clearly and suggest searching the knowledge base."
            )
            api_key = get_gemini_api_key()
            text_model, _ = get_models()
            client = genai.Client(api_key=api_key)
            try:
                resp = client.models.generate_content(
                    model=text_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
                text = (resp.text or "").strip()
                if text:
                    state["answer_markdown"] = _ensure_heading(text, plan_heading)
                    state["citations"] = []
                    return state
            except Exception:
                pass

        state["answer_markdown"] = _ensure_heading(
            "I don't have information about this in the available training manuals. "
            "Please try rephrasing your question or ask about a different topic.",
            "Information Not Available",
        )
        state["citations"] = []
        return state

    min_chunk_usage = min(len(hits), 3)
    
    # Calculate average similarity to assess evidence quality
    avg_similarity = sum(h.get('similarity', 0.0) for h in hits) / len(hits) if hits else 0.0
    evidence_quality = "strong" if avg_similarity >= 0.6 else "moderate" if avg_similarity >= 0.4 else "weak"

    evidence_lines: List[str] = []
    for idx, h in enumerate(hits, start=1):
        evidence_lines.append(
            f"[{idx}] (similarity: {h.get('similarity', 0.0):.2f})\n"
            f"{h.get('chunk_text', '')}\n"
        )

    # Build a simpler, more direct prompt
    response_style = "Answer directly and concisely."
    if mode['concise']:
        response_style = "Give a brief, to-the-point answer (2-4 sentences or 3-5 bullets max)."
    elif mode['detail']:
        response_style = "Provide a clear explanation with necessary details (1-2 short paragraphs)."
    elif mode['steps']:
        response_style = "List the steps clearly and concisely (numbered list)."
    elif mode['table']:
        response_style = "Present information in a comparison table."
    
    prompt = (
        "You are a Royal Navy training assistant. Answer the question using the evidence provided.\n\n"
        f"QUESTION: {user_query}\n\n"
        f"RESPONSE STYLE: {response_style}\n\n"
        f"EVIDENCE QUALITY: {evidence_quality}\\n\\n"
        "RULES:\\n"
        "- Start with ### heading that summarizes the answer\\n"
        "- Answer DIRECTLY - don't over-explain or add unnecessary context\\n"
        "- Use the evidence to support your answer\\n"
        "- Keep it SHORT and PRACTICAL\\n"
        "- Don't mention source files, pages, or citation numbers in your answer\\n"
        "- If evidence quality is weak or doesn't answer the question, be honest about limitations\\n"
        "- For strong evidence, answer confidently; for weak evidence, acknowledge uncertainty\\n\\n"
        f"CONTEXT: {book_hint or 'Royal Navy training manual'}\n\n"
        "EVIDENCE:\n"
        + "\n".join(evidence_lines)
        + "\n\nOUTPUT: Return JSON with keys 'answer_markdown' (string) and 'used_citations' (array of integers 1-"
        + str(len(hits))
        + "). No markdown fences."
    )

    api_key = get_gemini_api_key()
    text_model, _ = get_models()
    client = genai.Client(api_key=api_key)

    answer_markdown = ""
    used = []
    try:
        resp = client.models.generate_content(
            model=text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        data = parse_json_loose(resp.text or "")
        if isinstance(data, dict):
            ans = data.get("answer_markdown")
            if isinstance(ans, str) and ans.strip():
                answer_markdown = ans.strip()
            citations_raw = data.get("used_citations", [])
            if isinstance(citations_raw, list):
                used = [int(x) for x in citations_raw if isinstance(x, int) and 1 <= x <= len(hits)]
    except Exception:
        pass

    if not answer_markdown:
        top = hits[0]
        fallback_body = (top.get("answer") or top.get("chunk_text") or "")[:500].strip()
        answer_markdown = _ensure_heading(
            fallback_body or "Insufficient information found in the indexed documents.",
            plan_heading,
        )
        used = [1]

    if not used:
        used = [1]

    citations = []
    for idx in used:
        h = hits[idx - 1]
        citations.append(
            {
                "idx": idx,
                "source_file": h["source_file"],
                "page_start": h.get("page_start"),
                "line_start": h.get("line_start"),
            }
        )

    state["answer_markdown"] = _ensure_heading(answer_markdown, plan_heading)
    state["citations"] = citations
    return state
