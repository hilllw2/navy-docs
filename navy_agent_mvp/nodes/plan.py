from typing import List

_TABLE_HINTS = {"table", "compare", "comparison", "vs", "versus", "difference", "differences", "matrix"}
_STEP_HINTS = {"steps", "step", "procedure", "process", "checklist", "how to", "how do", "what do"}
_CONCISE_HINTS = {"summary", "summarize", "brief", "short", "key", "most important", "top", "quick"}
_DETAILED_HINTS = {"explain", "detailed", "detail", "why", "rationale", "background"}


def _detect_style_hints(question: str) -> dict:
    q = (question or "").lower()
    wants_table = any(h in q for h in _TABLE_HINTS)
    wants_steps = any(h in q for h in _STEP_HINTS)
    wants_concise = any(h in q for h in _CONCISE_HINTS)
    wants_detail = any(h in q for h in _DETAILED_HINTS)
    return {
        "table": wants_table,
        "steps": wants_steps,
        "concise": wants_concise and not wants_detail,
        "detail": wants_detail,
    }

from google import genai
from google.genai import types

from navy_agent_mvp.config import get_gemini_api_key, get_models
from navy_agent_mvp.state import AgentState, AnswerPlan
from navy_agent_mvp.utils import parse_json_loose, truncate


def _default_plan(user_query: str, book_hint: str = "") -> AnswerPlan:
    hints = _detect_style_hints(user_query)
    title = (user_query or "").strip().rstrip("?") or "Response"
    heading = title[:90]
    
    # Simplified: just one section for most cases
    sections = [{"title": "Answer", "instruction": "Provide a direct answer to the question."}]

    # Simple style tips based on question type
    style_tips = ["Be direct and concise."]
    if hints["steps"]:
        style_tips.append("Use numbered steps.")
    elif hints["table"]:
        style_tips.append("Use a comparison table.")
    elif hints["concise"]:
        style_tips.append("Keep it brief (3-5 bullets max).")

    return {
        "heading": heading,
        "sections": sections,
        "style_tips": style_tips,
    }


def plan_answer_node(state: AgentState) -> AgentState:
    user_query = state["user_query"]
    refined_query = state["route"].get("refined_query") or user_query
    hits = state.get("hits", [])
    book_hint = state.get("book_context_hint") or ""

    if not hits:
        state["answer_plan"] = _default_plan(user_query, book_hint)
        return state

    snippet_lines: List[str] = []
    for idx, hit in enumerate(hits[:5], start=1):
        snippet = truncate(hit.get("chunk_text") or "", 320)
        snippet_lines.append(
            f"[{idx}] source={hit.get('source_file')} page={hit.get('page_start')}\n{snippet}"
        )

    prompt = (
        "Create a simple answer plan for this naval training question.\n\n"
        f"QUESTION: {user_query}\n\n"
        "Return JSON with:\n"
        "- heading: Brief title for the answer (max 90 chars)\n"
        "- sections: Array with 1-2 objects, each having 'title' and 'instruction'\n"
        "- style_tips: Array with 1-2 brief style hints\n\n"
        "Keep it simple. For straightforward questions, use just one section.\n\n"
        f"EVIDENCE PREVIEW:\n" + "\n\n".join(snippet_lines[:3])
    )

    api_key = get_gemini_api_key()
    text_model, _ = get_models()
    client = genai.Client(api_key=api_key)

    plan = _default_plan(user_query, book_hint)
    try:
        resp = client.models.generate_content(
            model=text_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        data = parse_json_loose(resp.text or "")
        if isinstance(data, dict):
            heading = data.get("heading")
            if isinstance(heading, str) and heading.strip():
                plan["heading"] = heading.strip()[:90]

            sections = data.get("sections")
            parsed_sections: List[dict] = []
            if isinstance(sections, list):
                for section in sections:
                    if not isinstance(section, dict):
                        continue
                    title = section.get("title")
                    instruction = section.get("instruction")
                    if isinstance(title, str) and isinstance(instruction, str):
                        parsed_sections.append(
                            {
                                "title": title.strip()[:60] or "Section",
                                "instruction": instruction.strip()[:200] or "Explain the point.",
                            }
                        )
            if parsed_sections:
                plan["sections"] = parsed_sections[:4]

            style_tips = data.get("style_tips")
            parsed_tips: List[str] = []
            if isinstance(style_tips, list):
                for tip in style_tips:
                    if isinstance(tip, str) and tip.strip():
                        parsed_tips.append(tip.strip()[:120])
            if parsed_tips:
                plan["style_tips"] = parsed_tips[:3]
    except Exception:
        pass

    state["answer_plan"] = plan
    return state
