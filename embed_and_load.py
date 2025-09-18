# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 18:34:21 2025

@author: JIajie Shi
"""

# embed_and_load.py
import os, json, hashlib, pathlib, time
from typing import List, Dict

from dotenv import load_dotenv
from neo4j import GraphDatabase
import httpx

# ---- config ----
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j_password")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # pulled via `ollama pull ...`
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))              # 768 for nomic-embed-text

KB_PATH = pathlib.Path("kb.jsonl")
BATCH_SIZE = 100  # neo4j upsert batch size
TIMEOUT = 30.0

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using Ollama embeddings API.
    Uses one request per text (simple & robust).
    """
    out = []
    with httpx.Client(timeout=TIMEOUT) as client:
        for t in texts:
            # Ollama embeddings endpoint
            resp = client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": t},
            )
            resp.raise_for_status()
            data = resp.json()
            vec = data.get("embedding")
            if not vec or not isinstance(vec, list):
                raise RuntimeError("Invalid embedding response")
            out.append(vec)
            # be a little polite to local server
            time.sleep(0.01)
    return out

def ensure_indexes_and_constraints(tx):
    # Uniqueness
    tx.run("CREATE CONSTRAINT unique_page_url IF NOT EXISTS FOR (p:Page) REQUIRE p.url IS UNIQUE")
    tx.run("CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
    # Full-text (question + answer)
    tx.run("""
    CREATE FULLTEXT INDEX idx_fulltext_chunk IF NOT EXISTS
    FOR (n:Chunk) ON EACH [n.question, n.answer]
    """)
    # Vector index
    tx.run(f"""
    CREATE VECTOR INDEX idx_chunk_embedding IF NOT EXISTS
    FOR (n:Chunk) ON (n.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {EMBED_DIM},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """)

def upsert_batch(driver, rows: List[Dict], embeddings: List[List[float]]):
    """
    rows[i] corresponds to embeddings[i]
    """
    def work(tx, payload):
        tx.run("""
        UNWIND $rows AS row
        MERGE (p:Page {url: row.url})
          ON CREATE SET p.slug = row.slug
        MERGE (c:Chunk {id: row.id})
          ON CREATE SET c.kind = row.kind
        SET c.question = row.question,
            c.answer   = row.answer,
            c.url      = row.url,
            c.section  = row.section,
            c.source_format = row.source_format,
            c.vec_dim  = $vec_dim,
            c.embedding = row.embedding
        MERGE (c)-[:FROM_PAGE]->(p)
        """, rows=payload, vec_dim=EMBED_DIM)

    # Package rows with embeddings for Cypher
    payload = []
    for r, e in zip(rows, embeddings):
        payload.append({
            "id": r["id"],
            "kind": r["kind"],
            "question": r["question"],
            "answer": r["answer"],
            "url": r["url"],
            "slug": r.get("slug"),
            "section": r.get("section"),
            "source_format": r.get("source_format"),
            "embedding": e,
        })

    with driver.session() as sess:
        sess.execute_write(work, payload)

def slug_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
        parts = [x for x in (p.path or "/").split("/") if x]
        if parts and parts[0] == "tests":
            parts = parts[1:]
        return "/".join(parts)
    except Exception:
        return ""

def main():
    if not KB_PATH.exists():
        raise SystemExit("kb.jsonl not found. Run Phase 2 finalization first.")

    print(f"Using Neo4j: {NEO4J_URI}")
    print(f"Using Ollama: {OLLAMA_HOST}  model={EMBED_MODEL}  dim={EMBED_DIM}")

    # Load KB
    items = []
    with KB_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            # ensure required fields
            if not all(k in d for k in ("id","question","answer","url","kind")):
                continue
            # final text to embed: question + answer (grounded & rich)
            d["_embed_text"] = d["question"].strip() + "\n\n" + d["answer"].strip()
            d["slug"] = slug_from_url(d["url"])
            items.append(d)

    if not items:
        raise SystemExit("No items in kb.jsonl")

    # Connect Neo4j & create indexes
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as sess:
        sess.execute_write(ensure_indexes_and_constraints)
    print("[neo4j] indexes & constraints ensured")

    # Process in batches
    total = 0
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i:i+BATCH_SIZE]
        texts = [b["_embed_text"] for b in batch]
        vecs = embed_texts(texts)
        upsert_batch(driver, batch, vecs)
        total += len(batch)
        print(f"[load] upserted {total}/{len(items)}")

    driver.close()
    print("[done] All items loaded into Neo4j")

if __name__ == "__main__":
    main()
