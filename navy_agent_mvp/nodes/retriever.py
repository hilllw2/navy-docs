from typing import List, Optional

from google import genai
from google.genai import types

from navy_agent_mvp.config import EMBED_DIM, get_gemini_api_key, get_models, get_supabase_client
from navy_agent_mvp.state import AgentState
from navy_agent_mvp.utils import dedupe_hits, normalize_embedding, vector_literal


def _embed_query(query: str) -> List[float]:
    api_key = get_gemini_api_key()
    _, embed_model = get_models()
    client = genai.Client(api_key=api_key)

    result = client.models.embed_content(
        model=embed_model,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        ),
    )
    [emb] = result.embeddings
    values = list(emb.values)
    if len(values) != EMBED_DIM:
        raise ValueError(f"Expected embedding dim {EMBED_DIM}, got {len(values)}")
    return normalize_embedding(values)


def _rpc_search(query_embedding: List[float], top_k: int, source_file: Optional[str]):
    supabase = get_supabase_client()
    response = supabase.rpc(
        "match_naval_chunks",
        {
            "query_embedding": vector_literal(query_embedding),
            "match_count": top_k,
            "filter_source": source_file,
        },
    ).execute()
    return response.data or []


def retrieve_node(state: AgentState) -> AgentState:
    route = state["route"]
    refined_query = route["refined_query"]
    top_k = int(state.get("top_k", 6))
    confidence = float(route.get("routing_confidence", 0.0))
    target = route.get("target_source_file")
    source_file_lock = state.get("source_file_lock")

    qvec = _embed_query(refined_query)

    if source_file_lock:
        rows = _rpc_search(qvec, top_k, source_file_lock)
        mode = "filtered"
    elif target and confidence >= 0.7:
        filtered = _rpc_search(qvec, top_k, target)
        if len(filtered) >= max(3, top_k // 2):
            rows = filtered
            mode = "filtered"
        else:
            global_rows = _rpc_search(qvec, top_k, None)
            rows = dedupe_hits([*filtered, *global_rows])[:top_k]
            mode = "filtered_then_global"
    else:
        rows = _rpc_search(qvec, top_k, None)
        mode = "global"

    hits = []
    for row in rows:
        hits.append(
            {
                "id": str(row.get("id")),
                "source_file": row.get("source_file") or "",
                "page_start": row.get("page_start"),
                "line_start": row.get("line_start"),
                "chunk_text": row.get("chunk_text") or "",
                "question": row.get("question"),
                "answer": row.get("answer"),
                "similarity": float(row.get("similarity") or 0.0),
            }
        )

    state["hits"] = hits
    state["retrieval_mode"] = mode if hits else "none"
    return state
