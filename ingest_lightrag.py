# ingest_lightrag.py
import os, json, asyncio, hashlib, logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

load_dotenv(find_dotenv(usecwd=True), override=True)

WORKDIR     = os.getenv("LR_WORKDIR", "./lr_storage")
QA_FILE     = os.getenv("QA_FILE", "./qa.jsonl")   # your Phase-2 FAQ file
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL   = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5:7b-instruct-q4_K_M")

Path(WORKDIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_qas(path):
    items = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") != "qa":
                continue
            url = rec.get("url", "")
            q = (rec.get("question") or "").strip()
            a = (rec.get("answer")  or "").strip()

            # Keep URL in the text so we can extract Sources later
            text = f"### FAQ\nQ: {q}\nA: {a}\nSource: {url}"

            # Guaranteed-unique, stable ID (dedup exact same QA across runs)
            raw = f"{url}\nQ:{q}\nA:{a}"
            uid = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            _id = f"qa_{uid}"
            if _id in seen_ids:
                continue
            seen_ids.add(_id)
            items.append((_id, text))
    return items

async def main():
    print("Neo4j (LightRAG):", os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"),
          "PWD set?", bool(os.getenv("NEO4J_PASSWORD")))

    qa = load_qas(QA_FILE)
    logging.info("Loaded %d QA items from %s", len(qa), QA_FILE)
    if not qa:
        raise SystemExit("No QA items found in qa.jsonl")

    # LightRAG with Neo4j KG + Ollama embeddings/generation
    rag = LightRAG(
        working_dir=WORKDIR,
        graph_storage="Neo4JStorage",                 # KG lives in Neo4j
        llm_model_func=ollama_model_complete,         # used internally if needed
        llm_model_name=GEN_MODEL,
        llm_model_kwargs={"host": OLLAMA_HOST, "options": {"num_ctx": 8192}},
        embedding_func=EmbeddingFunc(
            embedding_dim=768,                        # nomic-embed-text dim
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=EMBED_MODEL, host=OLLAMA_HOST
            ),
        ),
        embedding_batch_num=16,          # NEW
        embedding_func_max_async=4,      # NEW
        llm_model_max_async=2,           # NEW
        #llm_model_max_token_size=8192,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    ids, texts = zip(*qa)
    logging.info("Inserting %d QAs into LightRAG…", len(texts))
    # If you prefer auto-ids, use: await rag.ainsert(list(texts))
    await rag.ainsert(list(texts), ids=list(ids))
    logging.info("Done. LightRAG workdir: %s", WORKDIR)

if __name__ == "__main__":
    asyncio.run(main())
