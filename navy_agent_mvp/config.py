import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from supabase import Client, create_client


EMBED_DIM = 2000


def load_env() -> None:
    """Load .env file locally. On Streamlit Cloud, env vars come from st.secrets."""
    load_dotenv(override=True)


def _secret(key: str, fallback: str = "") -> str:
    """Read from env var first, then from st.secrets (Streamlit Cloud), then fallback."""
    val = os.getenv(key, "").strip().strip("'\"")
    if val:
        return val
    try:
        import streamlit as st  # only available when running in Streamlit context
        secret_val = st.secrets.get(key, fallback)
        return str(secret_val).strip() if secret_val else fallback
    except Exception:
        return fallback


def get_models() -> tuple[str, str]:
    text_model = _secret("GEMINI_TEXT_MODEL") or "gemini-2.5-flash"
    embed_model = _secret("GEMINI_EMBED_MODEL") or "gemini-embedding-2-preview"
    return text_model, embed_model


def get_gemini_api_key() -> str:
    key = _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY — set it in .env locally or in Streamlit Cloud secrets.")
    return key


def get_supabase_client() -> Client:
    url = _secret("SUPABASE_URL")
    key = _secret("SUPABASE_ANON_KEY") or _secret("SUPABASE_KEY")
    if not url:
        raise RuntimeError("Missing SUPABASE_URL — set it in .env locally or in Streamlit Cloud secrets.")
    if not key:
        raise RuntimeError("Missing SUPABASE_ANON_KEY — set it in .env locally or in Streamlit Cloud secrets.")
    return create_client(url, key)


def load_book_catalog() -> List[Dict[str, Any]]:
    here = Path(__file__).resolve().parent
    path = here / "book_catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))
