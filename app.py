# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:39:56 2025

@author: JIajie Shi
"""

import os, json, math, re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from neo4j import GraphDatabase
import httpx
from lightrag_client import retrieve_context_with_sources

from fastapi.responses import HTMLResponse

# ------------------ Config ------------------
load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j_password")

OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Generation defaults
DEFAULT_GEN_BACKEND = os.getenv("DEFAULT_GEN_BACKEND", "ollama").lower()  # "ollama" or "openrouter"

# OpenRouter (optional)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")

# Ollama generation (local)
OLLAMA_GEN_MODEL   = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_NUM_CTX     = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# ------------------ FastAPI ------------------
app = FastAPI(title="Vibrant RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# Neo4j driver (one per process)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ------------------ Models ------------------
class AskRequest(BaseModel):
    query: str
    top_k: int = 6
    use_kg: bool = True
    gen_backend: Optional[str] = None  # "ollama" | "openrouter"
    temperature: float = 0.2

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# ------------------ Utils ------------------

async def generate_openrouter(messages: List[Dict[str, str]], temperature: float) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://vibrant-rag.local",
        "X-Title": "Vibrant-RAG-API",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 600,
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]

async def generate_ollama(messages: List[Dict[str, str]], temperature: float) -> str:
    payload = {
        "model": OLLAMA_GEN_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX}
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    return data.get("message", {}).get("content", "")

# ------------------ Routes ------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Vibrant RAG – Ask</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; margin:0; background:#0b1020; color:#e9edf5;}
    .wrap{max-width:900px; margin:40px auto; padding:24px;}
    h1{font-size:22px; margin:0 0 16px;}
    .card{background:#12182b; border:1px solid #1f2742; border-radius:14px; padding:16px; box-shadow:0 6px 24px rgba(0,0,0,.15);}
    textarea, input, select, button{width:100%; font:inherit; border-radius:10px; border:1px solid #2e385e; background:#0f1424; color:#e9edf5;}
    textarea{min-height:110px; padding:12px; resize:vertical;}
    .row{display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; margin-top:12px;}
    .row .col{display:flex; gap:8px; align-items:center;}
    label{font-size:13px; opacity:.85;}
    button{padding:12px; cursor:pointer; background:#2c7cf4; border-color:#2c7cf4; margin-top:12px;}
    button[disabled]{opacity:.6; cursor:not-allowed;}
    .answer{white-space:pre-wrap; background:#0f1424; border-radius:10px; padding:12px; border:1px solid #2e385e;}
    .sources a{color:#9ec2ff; text-decoration:none}
    .sources li{margin:4px 0}
    .muted{opacity:.8; font-size:12px}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Vibrant RAG – Ask a question</h1>
    <div class="card">
      <label for="q">Question</label>
      <textarea id="q" placeholder="e.g., Do I need to fast before taking the Food Sensitivity test?"></textarea>

      <div class="row">
        <div class="col">
          <label for="topk">Top K</label>
          <input id="topk" type="number" min="1" max="10" value="6"/>
        </div>
        <div class="col">
          <label><input id="usekg" type="checkbox" checked/> Use KG boost</label>
        </div>
        <div class="col">
          <label for="backend">Generator</label>
          <select id="backend">
            <option value="ollama" selected>Ollama (local)</option>
            <option value="openrouter">OpenRouter (cloud)</option>
          </select>
        </div>
      </div>

      <button id="askBtn">Ask</button>
      <p id="status" class="muted"></p>
    </div>

    <h2>Answer</h2>
    <div id="answer" class="answer">—</div>
    <h3>Sources</h3>
    <ul id="sources" class="sources"></ul>
  </div>

<script>
const qEl = document.getElementById('q');
const btn = document.getElementById('askBtn');
const ans = document.getElementById('answer');
const src = document.getElementById('sources');
const stat = document.getElementById('status');

async function ask() {
  const query = qEl.value.trim();
  const top_k = parseInt(document.getElementById('topk').value || '6', 10);
  const use_kg = document.getElementById('usekg').checked;
  const gen_backend = document.getElementById('backend').value;

  if (!query) { qEl.focus(); return; }

  btn.disabled = true; stat.textContent = 'Thinking…'; ans.textContent = '…'; src.innerHTML = '';
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ query, top_k, use_kg, gen_backend, temperature: 0.2 })
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    ans.textContent = data.answer || '(no answer)';
    (data.sources || []).forEach((s, i) => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = s.url; a.target = '_blank';
      a.textContent = `[${i+1}] ${s.question || s.url}`;
      li.appendChild(a);
      src.appendChild(li);
    });
    stat.textContent = '';
  } catch (e) {
    ans.textContent = 'Error: ' + (e.message || e);
    stat.textContent = 'Failed';
  } finally {
    btn.disabled = false;
  }
}

btn.addEventListener('click', ask);
qEl.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) { ask(); }
});
</script>
</body>
</html>
    """

@app.get("/health")
async def health():
    try:
        with driver.session() as s:
            s.run("RETURN 1 AS ok").single()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/config")
async def config():
    return {
        "neo4j_uri": NEO4J_URI,
        "embed_model": EMBED_MODEL,
        "gen_default": DEFAULT_GEN_BACKEND,
        "gen_ollama_model": OLLAMA_GEN_MODEL,
        "gen_openrouter_model": OPENROUTER_MODEL,
    }

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    backend = (req.gen_backend or DEFAULT_GEN_BACKEND).lower()

    # 1) Retrieve contexts with LightRAG (mix mode)
    ctx_text, urls = await retrieve_context_with_sources(req.query, top_k=req.top_k)

    # Ensure unique, ordered sources (max = top_k)
    seen = set()
    ordered_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered_urls.append(u)
        if len(ordered_urls) >= req.top_k:
            break

    # Build a numbered Sources block so the LLM can cite [1], [2], ...
    sources_block = "\n".join(f"[{i+1}] {u}" for i, u in enumerate(ordered_urls))

    # 2) Build prompt (ASK the model to use [n] citations)
    messages = [
        {"role": "system", "content":
         "You are a precise assistant for the Vibrant Wellness test menu. "
         "Answer ONLY using the provided context. If the answer is not in the context, say you don't know. "
         "Cite claims using bracketed numbers like [1], [2] that refer to the Sources list below. "
        },
        {"role": "user", "content":
         f"Question:\n{req.query}\n\n"
         f"Context:\n{ctx_text}\n\n"
         f"Sources:\n{sources_block}\n\n"
         f"Answer:"}
    ]

    # 3) Generate with your chosen backend (Ollama by default)
    answer = (await generate_ollama(messages, req.temperature)) if backend == "ollama" \
             else (await generate_openrouter(messages, req.temperature))

    # 4) Return answer + sources (keep numbering consistent with Sources block)
    srcs = [{"url": u, "question": ""} for u in ordered_urls]
    return AskResponse(answer=answer.strip(), sources=srcs)

