import argparse
import concurrent.futures
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from google.genai import types
from pypdf import PdfReader
from supabase import Client, create_client


EMBED_DIM = 2000


@dataclass
class Chunk:
    source_file: str
    page_start: int
    page_end: int
    chunk_index: int
    line_start: Optional[int]
    chunk_text: str
    question: Optional[str]
    answer: Optional[str]
    embedding: List[float]


def estimate_tokens(text: str) -> int:
    # Cheap heuristic for planning chunk size.
    words = len(text.split())
    return max(1, math.ceil(words * 1.3))


def normalize_embedding(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def get_supabase_client() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    if not supabase_url:
        raise RuntimeError("Missing SUPABASE_URL")
    if not supabase_anon_key:
        raise RuntimeError("Missing SUPABASE_ANON_KEY (or SUPABASE_KEY)")
    return create_client(supabase_url, supabase_anon_key)


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Simple sentence split that handles common punctuation.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def parse_page_paragraphs(page_text: str) -> List[tuple[str, Optional[int]]]:
    lines = page_text.splitlines()
    paragraphs: List[tuple[str, Optional[int]]] = []

    current: List[str] = []
    start_line: Optional[int] = None

    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append((" ".join(current).strip(), start_line))
                current = []
                start_line = None
            continue

        if start_line is None:
            start_line = i
        current.append(line)

    if current:
        paragraphs.append((" ".join(current).strip(), start_line))

    return [(p, ln) for p, ln in paragraphs if p]


def chunk_paragraphs(
    paragraphs: List[tuple[str, Optional[int]]],
    target_tokens: int,
    overlap_tokens: int,
) -> List[tuple[str, Optional[int]]]:
    chunks: List[tuple[str, Optional[int]]] = []
    buffer: List[str] = []
    buffer_start: Optional[int] = None

    def flush() -> None:
        nonlocal buffer, buffer_start
        if not buffer:
            return

        combined = "\n\n".join(buffer).strip()
        if not combined:
            buffer = []
            buffer_start = None
            return

        chunks.append((combined, buffer_start))

        # Overlap by trailing sentences from this chunk.
        if overlap_tokens <= 0:
            buffer = []
            buffer_start = None
            return

        sentences = split_sentences(combined)
        tail: List[str] = []
        tail_tokens = 0
        for s in reversed(sentences):
            s_tokens = estimate_tokens(s)
            if tail and (tail_tokens + s_tokens) > overlap_tokens:
                break
            tail.append(s)
            tail_tokens += s_tokens
        tail.reverse()

        if tail:
            overlap_text = " ".join(tail)
            buffer = [overlap_text]
        else:
            buffer = []
        # Keep same line start for overlap continuation.

    for paragraph, line_start in paragraphs:
        p_tokens = estimate_tokens(paragraph)

        # If one paragraph is giant, sentence-split it.
        if p_tokens > target_tokens * 1.3:
            for sentence in split_sentences(paragraph):
                if not buffer:
                    buffer_start = line_start
                candidate = "\n\n".join(buffer + [sentence]).strip()
                if buffer and estimate_tokens(candidate) > target_tokens:
                    flush()
                    if not buffer:
                        buffer_start = line_start
                buffer.append(sentence)
            continue

        if not buffer:
            buffer_start = line_start

        candidate = "\n\n".join(buffer + [paragraph]).strip()
        if buffer and estimate_tokens(candidate) > target_tokens:
            flush()
            if not buffer:
                buffer_start = line_start

        buffer.append(paragraph)

    flush()
    return chunks


def extract_chunks_from_pdf(
    pdf_path: Path,
    target_tokens: int,
    overlap_tokens: int,
) -> List[tuple[int, int, Optional[int], str]]:
    reader = PdfReader(str(pdf_path))
    out: List[tuple[int, int, Optional[int], str]] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue

        paragraphs = parse_page_paragraphs(text)
        if not paragraphs:
            continue

        chunks = chunk_paragraphs(paragraphs, target_tokens=target_tokens, overlap_tokens=overlap_tokens)

        for chunk_text, line_start in chunks:
            if chunk_text.strip():
                out.append((page_idx, page_idx, line_start, chunk_text.strip()))

    return out


def parse_json_loose(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try first JSON array/object in the text.
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_qa_heuristic(chunk_text: str) -> tuple[Optional[str], Optional[str]]:
    clean = re.sub(r"\s+", " ", chunk_text).strip()
    if not clean:
        return None, None

    sentences = split_sentences(clean)
    if not sentences:
        short = clean[:220].strip()
        return (f"What does this section say about {short[:80]}?", short) if short else (None, None)

    first = sentences[0].strip()
    answer = " ".join(sentences[:2]).strip()

    heading_match = re.match(r"^([A-Z][A-Z0-9 /,()\-]{4,80})$", first)
    if heading_match:
        topic = heading_match.group(1).strip().title()
        question = f"What does this section say about {topic}?"
        if len(sentences) > 1:
            answer = " ".join(sentences[1:3]).strip() or answer
    else:
        topic = re.sub(r"^[Tt]he\s+", "", first)
        topic = re.sub(r"[.?!].*$", "", topic).strip()
        topic = topic[:90] if topic else "this topic"
        question = f"What does the document say about {topic}?"

    return question[:200], answer[:500] if answer else None


def build_qa_batch_prompt(chunk_texts: Sequence[str]) -> str:
    parts = [
        "You are creating retrieval metadata for PDF chunks.",
        "Return STRICT JSON as an array.",
        "Each item must have exactly these keys: chunk_index, question, answer.",
        "- chunk_index: integer matching the input chunk index.",
        "- question: one realistic user question this chunk can answer.",
        "- answer: concise factual answer grounded only in the chunk.",
        "- If a chunk is noisy or unusable, return null for question and answer.",
        "No markdown. No prose. JSON only.",
        "",
    ]
    for idx, chunk_text in enumerate(chunk_texts):
        parts.append(f"CHUNK_INDEX: {idx}\nCHUNK_TEXT:\n{chunk_text}\n")
    return "\n".join(parts)


def sanitize_qa_value(value: Any, max_len: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned[:max_len] if cleaned else None


def generate_qa_gemini_batch(
    api_key: str,
    text_model: str,
    chunk_texts: Sequence[str],
) -> List[tuple[Optional[str], Optional[str]]]:
    client = genai.Client(api_key=api_key)
    prompt = build_qa_batch_prompt(chunk_texts)
    prompt = (
        prompt
    )

    max_attempts = 5
    resp = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.models.generate_content(
                model=text_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            break
        except ClientError as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            if status == 429 and attempt < max_attempts:
                sleep_s = min(90, 10 * attempt)
                print(f"Q/A batch rate-limited; retrying in {sleep_s}s (attempt {attempt}/{max_attempts})")
                time.sleep(sleep_s)
                continue
            return [generate_qa_heuristic(text) for text in chunk_texts]
        except Exception:
            return [generate_qa_heuristic(text) for text in chunk_texts]

    if resp is None:
        return [generate_qa_heuristic(text) for text in chunk_texts]

    data = parse_json_loose(resp.text or "")
    if not isinstance(data, list):
        return [generate_qa_heuristic(text) for text in chunk_texts]

    results: List[tuple[Optional[str], Optional[str]]] = [
        (None, None) for _ in chunk_texts
    ]
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = item.get("chunk_index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(chunk_texts):
            continue
        question = sanitize_qa_value(item.get("question"), 200)
        answer = sanitize_qa_value(item.get("answer"), 500)
        results[idx] = (question, answer)

    for idx, pair in enumerate(results):
        if pair == (None, None):
            results[idx] = generate_qa_heuristic(chunk_texts[idx])
    return results


def generate_qa(
    client: genai.Client,
    text_model: str,
    chunk_text: str,
    qa_mode: str,
) -> tuple[Optional[str], Optional[str]]:
    if qa_mode == "heuristic":
        return generate_qa_heuristic(chunk_text)
    return generate_qa_gemini_batch("", text_model, [chunk_text])[0]


def fill_chunk_qas(
    chunks: List[Chunk],
    api_key: str,
    text_model: str,
    qa_mode: str,
    qa_batch_size: int,
    qa_workers: int,
) -> None:
    if not chunks:
        return

    if qa_mode == "heuristic":
        for chunk in chunks:
            chunk.question, chunk.answer = generate_qa_heuristic(chunk.chunk_text)
        return

    indexed_batches = list(enumerate(batched(chunks, qa_batch_size), start=1))

    def process_batch(batch_info: tuple[int, Sequence[Chunk]]) -> tuple[int, List[tuple[Optional[str], Optional[str]]]]:
        batch_number, batch_chunks = batch_info
        print(f"Generating Q/A batch {batch_number}/{len(indexed_batches)} ({len(batch_chunks)} chunks)")
        qas = generate_qa_gemini_batch(
            api_key=api_key,
            text_model=text_model,
            chunk_texts=[chunk.chunk_text for chunk in batch_chunks],
        )
        return batch_number, qas

    max_workers = max(1, qa_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch_info): batch_info
            for batch_info in indexed_batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_number, batch_chunks = future_to_batch[future]
            qas = future.result()[1]
            for chunk, (question, answer) in zip(batch_chunks, qas):
                chunk.question = question
                chunk.answer = answer
            print(f"Finished Q/A batch {batch_number}/{len(indexed_batches)}")


def embed_texts(
    client: genai.Client,
    embed_model: str,
    texts: Sequence[str],
    task_type: str,
) -> List[List[float]]:
    if not texts:
        return []

    result = client.models.embed_content(
        model=embed_model,
        contents=list(texts),
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        ),
    )

    vectors: List[List[float]] = []
    for e in result.embeddings:
        values = list(e.values)
        if len(values) != EMBED_DIM:
            raise ValueError(f"Expected embedding dim {EMBED_DIM}, got {len(values)}")
        vectors.append(normalize_embedding(values))

    return vectors


def batched(items: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def ingest_pdf(
    pdf_path: Path,
    target_tokens: int,
    overlap_tokens: int,
    batch_size: int,
    qa_mode: str,
    qa_batch_size: int,
    qa_workers: int,
) -> None:
    load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY")
    text_model = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
    embed_model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    supabase = get_supabase_client()

    raw_chunks = extract_chunks_from_pdf(pdf_path, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    if not raw_chunks:
        print("No extractable text chunks found.")
        return

    print(f"Extracted chunks: {len(raw_chunks)}")

    source_name = pdf_path.name

    assembled: List[Chunk] = []
    for idx, (page_start, page_end, line_start, chunk_text) in enumerate(raw_chunks):
        assembled.append(
            Chunk(
                source_file=source_name,
                page_start=page_start,
                page_end=page_end,
                chunk_index=idx,
                line_start=line_start,
                chunk_text=chunk_text,
                question=None,
                answer=None,
                embedding=[],
            )
        )

    fill_chunk_qas(
        chunks=assembled,
        api_key=api_key,
        text_model=text_model,
        qa_mode=qa_mode,
        qa_batch_size=qa_batch_size,
        qa_workers=qa_workers,
    )

    # Batch embed for throughput.
    for batch in batched(assembled, batch_size):
        texts = [c.chunk_text for c in batch]
        vectors = embed_texts(
            client,
            embed_model=embed_model,
            texts=texts,
            task_type="RETRIEVAL_DOCUMENT",
        )
        for c, vec in zip(batch, vectors):
            c.embedding = vec

    for batch in batched(assembled, batch_size):
        rows = [
            {
                "source_file": c.source_file,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_index": c.chunk_index,
                "line_start": c.line_start,
                "chunk_text": c.chunk_text,
                "question": c.question,
                "answer": c.answer,
                "embedding": vector_literal(c.embedding),
            }
            for c in batch
        ]
        supabase.table("naval_doc_chunks").insert(rows).execute()

    print(f"Inserted rows: {len(assembled)}")


def search(
    query: str,
    top_k: int,
    source_file: Optional[str],
) -> None:
    load_dotenv(override=True)

    api_key = os.getenv("GEMINI_API_KEY")
    embed_model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    supabase = get_supabase_client()

    [query_vec] = embed_texts(
        client,
        embed_model=embed_model,
        texts=[query],
        task_type="RETRIEVAL_QUERY",
    )

    response = supabase.rpc(
        "match_naval_chunks",
        {
            "query_embedding": vector_literal(query_vec),
            "match_count": top_k,
            "filter_source": source_file,
        },
    ).execute()
    rows = response.data or []

    if not rows:
        print("No matches found.")
        return

    for i, row in enumerate(rows, start=1):
        src = row.get("source_file")
        page = row.get("page_start")
        line = row.get("line_start")
        chunk_text = row.get("chunk_text") or ""
        question = row.get("question")
        answer = row.get("answer")
        similarity = float(row.get("similarity") or 0.0)
        print("-" * 80)
        print(f"#{i} | sim={similarity:.4f} | source={src} | page={page} | line={line}")
        if question:
            print(f"Q: {question}")
        if answer:
            print(f"A: {answer}")
        print(f"Chunk: {chunk_text[:600]}{'...' if len(chunk_text) > 600 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF -> Q/A -> Gemini embeddings -> Postgres")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into naval_doc_chunks")
    p_ingest.add_argument("--pdf", required=True, help="Absolute path to PDF")
    p_ingest.add_argument("--target-tokens", type=int, default=700)
    p_ingest.add_argument("--overlap-tokens", type=int, default=100)
    p_ingest.add_argument("--batch-size", type=int, default=64)
    p_ingest.add_argument("--qa-mode", choices=["heuristic", "gemini"], default="gemini")
    p_ingest.add_argument("--qa-batch-size", type=int, default=10)
    p_ingest.add_argument("--qa-workers", type=int, default=2)

    p_search = sub.add_parser("search", help="Search indexed chunks")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--source-file", default=None)

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_pdf(
            pdf_path=Path(args.pdf).expanduser().resolve(),
            target_tokens=args.target_tokens,
            overlap_tokens=args.overlap_tokens,
            batch_size=args.batch_size,
            qa_mode=args.qa_mode,
            qa_batch_size=args.qa_batch_size,
            qa_workers=args.qa_workers,
        )
    elif args.cmd == "search":
        search(
            query=args.query,
            top_k=args.top_k,
            source_file=args.source_file,
        )


if __name__ == "__main__":
    main()
