# build_kg.py
# Phase 3b: Build a Knowledge Graph from kb.jsonl using a local Ollama model (or OpenRouter).
# - Batches multiple chunks per LLM call
# - Caches results per chunk
# - Writes Entities/Relations/MENTIONS into Neo4j
# - Prints clear progress (processed / total, upserted rows)

import os, json, pathlib, re, time, hashlib, random
from typing import Dict, List, Any

import httpx
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ------------------ Config ------------------
load_dotenv()

# Backend: "ollama" (default) or "openrouter"
KG_BACKEND      = os.getenv("KG_BACKEND", "ollama").lower()

# Ollama
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_KG_MODEL = os.getenv("OLLAMA_KG_MODEL", "qwen2.5:7b-instruct-q4_K_M")
OLLAMA_NUM_CTX  = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# OpenRouter (only if you switch KG_BACKEND=openrouter)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
RATE_LIMIT_S       = float(os.getenv("KG_RATE_LIMIT_S", "3.5"))
MAX_WAIT_429       = float(os.getenv("KG_MAX_WAIT_ON_429", "90"))

# Neo4j
NEO4J_URI      = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j_password")

KB_PATH = pathlib.Path("kb.jsonl")

# Tuning knobs
BATCH_UPSERT   = int(os.getenv("KG_BATCH_UPSERT", "50"))
MAX_TRIPLES    = int(os.getenv("KG_MAX_TRIPLES", "8"))
DRY_LIMIT      = os.getenv("KG_DRY_LIMIT")                # e.g. "120"
PROCESS_KIND   = os.getenv("KG_PROCESS_KIND", "qa")       # "qa" or "all"
BUNDLE_SIZE    = int(os.getenv("KG_BUNDLE_SIZE", "6"))    # items per LLM call

CACHE_DIR = pathlib.Path("kg_cache"); CACHE_DIR.mkdir(exist_ok=True)

PROMPT_SYS = (
    "You are an information extraction assistant for clinical laboratory test pages. "
    "Extract a compact knowledge graph in JSON capturing key domain entities and relations. "
    "Be faithful to the text. Use concise, domain-specific predicates."
)

PROMPT_USER_TEMPLATE_MULTI = """For EACH item below, extract entities and subject–predicate–object triples.

Rules:
- Return ONLY valid JSON: a list of objects, one per input, each with keys:
  "id", "entities", "triples".
- "entities": list of objects {{"name": str, "type": str, "aliases": [str]?}}
  Suggested types: Test, Analyte, Biomarker, Condition, Symptom, SampleType, Method,
  Preparation, Contraindication, TurnaroundTime, Population, Organism, Panel, Instrument.
- "triples": list of objects {{"subj": str, "predicate": str, "obj": str}}
  Use short, domain verbs/noun phrases: "measures", "requires", "sample_type", "method",
  "turnaround_time", "includes", "assesses", "used_for", "detects".
- Limit to at most {max_triples} triples per item.
- Use surface forms present in the text. Do NOT invent facts.

Input items (array; each has 'id' and 'text'):
{payload_json}
"""

# ------------------ Helpers ------------------
def require_env():
    if not KB_PATH.exists():
        raise SystemExit("kb.jsonl not found. Run Phase 2 finalization first.")
    if KG_BACKEND == "openrouter" and not OPENROUTER_API_KEY:
        raise SystemExit("Set OPENROUTER_API_KEY in .env or use KG_BACKEND=ollama")

def canonical_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[\u2010-\u2015\-]+", "-", s)           # normalize hyphens
    s = re.sub(r"[^a-z0-9\s\-\+\&/]", "", s)            # keep simple chars
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_kb() -> List[Dict[str, Any]]:
    items = []
    with KB_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            if not all(k in d for k in ("id","question","answer","url","kind")):
                continue
            d["_text"] = d["question"].strip() + "\n\n" + d["answer"].strip()
            items.append(d)
    # Prefer QA first (high-signal), or only QA if requested
    if PROCESS_KIND.lower() == "qa":
        items = [x for x in items if x.get("kind") == "qa"]
    else:
        items.sort(key=lambda x: 0 if x.get("kind") == "qa" else 1)
    if DRY_LIMIT and DRY_LIMIT.isdigit():
        items = items[:int(DRY_LIMIT)]
    return items

def cache_path_for_id(chunk_id: str) -> pathlib.Path:
    return CACHE_DIR / f"{chunk_id}.json"

def have_cache(chunk_id: str) -> bool:
    return cache_path_for_id(chunk_id).exists()

def write_cache(chunk_id: str, data: Dict[str, Any]) -> None:
    cache_path_for_id(chunk_id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def read_cache(chunk_id: str) -> Dict[str, Any]:
    return json.loads(cache_path_for_id(chunk_id).read_text(encoding="utf-8"))

def normalize_extraction(ex: Dict[str, Any]) -> Dict[str, Any]:
    ents = []
    seen_c = set()
    for e in ex.get("entities", []) or []:
        name = (e.get("name") or "").strip()
        if not name: continue
        canon = canonical_name(name)
        if not canon or canon in seen_c: continue
        seen_c.add(canon)
        ents.append({
            "name": name,
            "canon": canon,
            "types": [ (e.get("type") or "").strip() ] if e.get("type") else [],
            "aliases": [a.strip() for a in (e.get("aliases") or []) if a and a.strip()],
        })
    triples = []
    for t in ex.get("triples", []) or []:
        s = (t.get("subj") or "").strip()
        p = (t.get("predicate") or "").strip()
        o = (t.get("obj") or "").strip()
        if not (s and p and o): continue
        triples.append({
            "subj": s, "subj_canon": canonical_name(s),
            "predicate": p.lower().strip(),
            "obj": o, "obj_canon": canonical_name(o),
        })
    return {"entities": ents, "triples": triples}

def parse_json_forgiving(text: str):
    # Try direct JSON; then fenced; then first [] block
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip code fences
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.S|re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # first bracketed JSON array
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    return []

# ------------------ LLM calls ------------------
def ollama_call_multi(items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Local LLM via Ollama /api/chat. items = [{id, text}, ...]
    """
    url = f"{OLLAMA_HOST}/api/chat"
    user_prompt = PROMPT_USER_TEMPLATE_MULTI.format(
        payload_json=json.dumps(items, ensure_ascii=False),
        max_triples=MAX_TRIPLES,
    )
    payload = {
        "model": OLLAMA_KG_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"num_ctx": OLLAMA_NUM_CTX}
    }
    with httpx.Client(timeout=240.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        return parse_json_forgiving(content)

def openrouter_call_multi(items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://vibrant-rag.local",
        "X-Title": "Vibrant-RAG-KG",
    }
    prompt_user = PROMPT_USER_TEMPLATE_MULTI.format(
        payload_json=json.dumps(items, ensure_ascii=False),
        max_triples=MAX_TRIPLES,
    )
    body = {
        "model": OPENROUTER_MODEL,
        "temperature": 0.1,
        "max_tokens": 1400,
        "messages": [
            {"role": "system", "content": PROMPT_SYS},
            {"role": "user", "content": prompt_user},
        ],
    }

    backoff = 2.0
    with httpx.Client(timeout=120.0) as client:
        while True:
            r = client.post(OPENROUTER_URL, headers=headers, json=body)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                data = parse_json_forgiving(content)
                time.sleep(max(RATE_LIMIT_S, 3.5))
                return data

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                reset_ms = r.headers.get("X-RateLimit-Reset")
                wait = None
                if reset_ms:
                    try:
                        wait = max(0.0, (int(reset_ms) / 1000.0) - time.time())
                    except Exception:
                        wait = None
                if wait is None and retry_after:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = None
                if wait is None:
                    wait = backoff + random.uniform(0,0.5)
                    backoff = min(backoff * 2, 60.0)
                if wait > float(os.getenv("KG_MAX_WAIT_ON_429", "90")):
                    raise RuntimeError(f"429 rate-limited; suggested wait {wait:.1f}s exceeds limit.")
                print(f"[429] rate-limited; sleeping {wait:.1f}s…")
                time.sleep(wait)
                continue

            if r.status_code in (502,503,504,408):
                print(f"[{r.status_code}] transient; retrying in {backoff:.1f}s…")
                time.sleep(backoff + random.uniform(0,0.5))
                backoff = min(backoff * 2, 60.0)
                continue

            try:
                print("[openrouter error]", r.status_code, r.text[:400])
            except Exception:
                pass
            r.raise_for_status()

def extract_multi(items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    return ollama_call_multi(items) if KG_BACKEND == "ollama" else openrouter_call_multi(items)

# ------------------ Neo4j ------------------
def ensure_kg_indexes(driver):
    with driver.session() as s:
        s.run("CREATE CONSTRAINT unique_entity_canon IF NOT EXISTS FOR (e:Entity) REQUIRE e.canon IS UNIQUE")
        s.run("""
        CREATE FULLTEXT INDEX idx_entity_names IF NOT EXISTS
        FOR (e:Entity) ON EACH [e.name, e.aliases]
        """)
        s.run("CREATE INDEX idx_entity_types IF NOT EXISTS FOR (e:Entity) ON (e.types)")

def upsert_kg(driver, rows: List[Dict[str, Any]]):
    cypher = """
    UNWIND $rows AS row
    // Upsert entities
    UNWIND row.entities AS e
    MERGE (ent:Entity {canon: e.canon})
      ON CREATE SET ent.name = e.name, ent.types = e.types, ent.aliases = e.aliases
      ON MATCH  SET ent.name = coalesce(ent.name, e.name)
    WITH row
    MATCH (c:Chunk {id: row.chunk_id})
    WITH row, c
    // Mentions
    UNWIND row.entities AS e2
    MATCH (ent2:Entity {canon: e2.canon})
    MERGE (c)-[:MENTIONS]->(ent2)
    WITH row
    // Relations with provenance
    UNWIND row.triples AS t
    MATCH (s:Entity {canon: t.subj_canon})
    MATCH (o:Entity {canon: t.obj_canon})
    MERGE (s)-[r:REL {predicate: t.predicate}]->(o)
    SET r.support_count = coalesce(r.support_count, 0) + 1,
        r.source_chunks = CASE
          WHEN r.source_chunks IS NULL THEN [row.chunk_id]
          WHEN NOT row.chunk_id IN r.source_chunks THEN r.source_chunks + row.chunk_id
          ELSE r.source_chunks
        END
    """
    with driver.session() as sess:
        sess.run(cypher, rows=rows)

# ------------------ Main ------------------
def main():
    require_env()
    items_all = load_kb()
    total = len(items_all)
    if total == 0:
        print("No items to process.")
        return

    print(f"[KG] backend={KG_BACKEND}"
          f" | model={(OLLAMA_KG_MODEL if KG_BACKEND=='ollama' else OPENROUTER_MODEL)}"
          f" | total={total} | bundle={BUNDLE_SIZE} | triples≤{MAX_TRIPLES}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    ensure_kg_indexes(driver)

    # Skip items already cached
    todo = [it for it in items_all if not have_cache(it["id"])]
    already = total - len(todo)
    if already:
        print(f"[cache] {already}/{total} items already cached; will process {len(todo)} new")

    processed = already
    upserted_rows_total = 0
    batch_rows: List[Dict[str, Any]] = []

    i = 0
    while i < len(todo):
        bundle = todo[i:i+BUNDLE_SIZE]
        payload = [{"id": it["id"], "text": it["_text"]} for it in bundle]

        try:
            results = extract_multi(payload)  # [{id, entities, triples}, ...]
        except Exception as e:
            print(f"[warn] extraction failed for bundle starting {bundle[0]['id']}: {e}")
            break

        # map results by id
        res_by_id = {r.get("id"): r for r in (results or []) if r and r.get("id")}
        for it in bundle:
            cid = it["id"]
            r = res_by_id.get(cid, {"entities": [], "triples": []})
            norm = normalize_extraction(r)
            write_cache(cid, norm)
            processed += 1
            if norm["entities"] or norm["triples"]:
                batch_rows.append({"chunk_id": cid,
                                   "entities": norm["entities"],
                                   "triples":  norm["triples"]})
        pct = processed / total * 100
        print(f"[progress] processed {processed}/{total} ({pct:.1f}%), queued {len(batch_rows)}")

        if len(batch_rows) >= BATCH_UPSERT:
            upsert_kg(driver, batch_rows)
            upserted_rows_total += len(batch_rows)
            print(f"[neo4j] upserted rows: {upserted_rows_total} (≤ processed {processed})")
            batch_rows = []

        i += BUNDLE_SIZE

    if batch_rows:
        upsert_kg(driver, batch_rows)
        upserted_rows_total += len(batch_rows)
        print(f"[neo4j] upserted rows: {upserted_rows_total} (final)")

    driver.close()
    print(f"[done] Processed {processed}/{total}. Upserted rows: {upserted_rows_total}. Cache: {CACHE_DIR.resolve()}")

if __name__ == "__main__":
    main()
