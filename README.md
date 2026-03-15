# Naval PDF -> Q/A -> Embeddings (MVP)

This MVP ingests a PDF, performs semantic-ish chunking, generates one Q/A per chunk, creates **Gemini embeddings with 2000 dimensions**, and inserts into your existing `naval_doc_chunks` table.

## Setup

1. Create venv and install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Copy env:
   - `cp .env.example .env`
   - Fill API key + Supabase URL + anon key.

## Ingest a PDF

`python mvp_rag.py ingest --pdf /absolute/path/to/file.pdf`

Optional flags:
- `--target-tokens 700`
- `--overlap-tokens 100`
- `--batch-size 100`

## Search

`python mvp_rag.py search --query "what is damage control" --top-k 5`

Optional:
- `--source-file "my_manual.pdf"`

## Notes

- Uses Gemini embeddings model with `output_dimensionality=2000` (required).
- Embeddings are normalized before storage/query for better cosine behavior.
- DB schema is unchanged.
- Uses Supabase Python SDK (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) for inserts and RPC search.

## LangGraph + Streamlit MVP (Routing + Q&A)

This project also includes a multi-node LangGraph app in [navy_agent_mvp/app.py](navy_agent_mvp/app.py):
- Router node: outputs refined query + best matching `source_file`.
- Retriever node: calls `match_naval_chunks` with source filter fallback logic.
- Answer node: returns grounded answer with citations.
- Explain node: returns evidence cards for "why this answer" UI.

Run UI:

`streamlit run navy_agent_mvp/app.py`

The catalog used by the router is in [navy_agent_mvp/book_catalog.json](navy_agent_mvp/book_catalog.json).
