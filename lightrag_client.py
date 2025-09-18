# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 00:56:00 2025

@author: JIajie Shi
"""

# lightrag_client.py
import os, re, asyncio
from dotenv import load_dotenv, find_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

load_dotenv(find_dotenv(usecwd=True), override=True)

WORKDIR     = os.getenv("LR_WORKDIR", "./lr_storage")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL   = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5:7b-instruct-q4_K_M")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))

_rag = None

async def get_rag():
    global _rag
    if _rag is None:
        _rag = LightRAG(
            working_dir=WORKDIR,
            graph_storage="Neo4JStorage",
            llm_model_func=ollama_model_complete,
            llm_model_name=GEN_MODEL,
            llm_model_kwargs={"host": OLLAMA_HOST, "options": {"num_ctx": 32768}},
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts, embed_model=EMBED_MODEL, host=OLLAMA_HOST
                ),
            ),
        )
        await _rag.initialize_storages()
        await initialize_pipeline_status()
    return _rag

def _parse_sources(ctx_text: str, max_items: int = 10):
    urls = []
    # 1) Prefer explicit "Source: ..." lines we added at ingest
    urls += re.findall(r"Source:\s*(https?://[^\s\]\)]+)", ctx_text)

    # 2) Fallback: grab any bare URLs in the context
    if not urls:
        urls += re.findall(r"https?://[^\s\]\)]+", ctx_text)

    # Normalize + de-dupe while preserving order
    seen = set()
    out = []
    for u in urls:
        u = u.rstrip(").,]")   # trim common trailing punctuation
        if u and u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= max_items:
            break
    return out

async def retrieve_context_with_sources(query: str, top_k: int = 8):
    rag = await get_rag()
    param = QueryParam(mode="mix", only_need_context=True)

    raw_ctx = await rag.aquery(query, param=param)

    # Extract URLs from the full, untrimmed text
    urls = _parse_sources(raw_ctx, max_items=top_k)

    # Now trim the context for the prompt
    ctx_text = raw_ctx[:MAX_CONTEXT_CHARS]
    return ctx_text, urls
