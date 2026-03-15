import json
import re
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np


def normalize_embedding(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def parse_json_loose(text: str) -> Optional[Any]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def dedupe_hits(rows: Iterable[dict]) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for row in rows:
        key = (
            row.get("source_file"),
            row.get("page_start"),
            (row.get("chunk_text") or "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
