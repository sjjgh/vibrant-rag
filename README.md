AI-Powered FAQ Answering System with Knowledge Graph (Vibrant Wellness)


1) Brief Project Description
----------------------------
This project builds a Retrieval-Augmented Generation (RAG) system that answers questions using only the
Vibrant Wellness test menu content. It combines:
- Crawl & parse of test pages (crawl4ai)
- A knowledge graph (Neo4j) and LightRAG for mixed retrieval + reranking (vector + keyword + KG)
- Local embeddings and generation with Ollama (no cloud costs/rate limits)
- A small FastAPI app with a one-page UI to ask questions and see answers with sources

The pipeline: crawl → parse FAQs → ingest into LightRAG (for retrieval) → query via FastAPI → generate answer with citations.


2) Environment & Dependencies (Windows + Anaconda)
--------------------------------------------------
Prerequisites
- Anaconda (Python 3.10+ environment)
- Git (optional, for repo)
- Neo4j Desktop (start a local DB: user neo4j, set a password)
- Ollama (install from https://ollama.com/download and keep it running)

Models (pull once in a terminal)
    ollama pull nomic-embed-text
    ollama pull qwen2.5:7b-instruct-q4_K_M

Create & activate the conda environment
    conda create -n vibrant-rag python=3.10 -y
    conda activate vibrant-rag

Install Python dependencies
    python -m pip install -r requirements.txt
    # (If you don't have a requirements.txt, install directly:)
    # python -m pip install fastapi uvicorn[standard] python-dotenv httpx neo4j crawl4ai playwright lightrag-hku[api]

Playwright browser dependencies (needed by crawl4ai once):
    playwright install

Environment variables
- Copy .env.example to .env and fill values:
    NEO4J_URI=neo4j://localhost:7687
    NEO4J_USER=neo4j                 # used by your own scripts (build_kg.py, etc.)
    NEO4J_PASSWORD=YOUR_PASSWORD
    NEO4J_USERNAME=neo4j             # LightRAG expects USERNAME (not USER)
    NEO4J_PASSWORD=YOUR_PASSWORD     # reused for LightRAG
    OLLAMA_HOST=http://localhost:11434
    EMBED_MODEL=nomic-embed-text
    OLLAMA_GEN_MODEL=qwen2.5:7b-instruct-q4_K_M
    OLLAMA_NUM_CTX=4096
    LR_WORKDIR=./lr_storage
    MAX_CONTEXT_CHARS=4000

Notes
- Keep your real .env out of version control. Use .env.example for sharing.
- Ensure your Neo4j DB is running before ingestion/query steps.


3) Step-by-Step: Data Ingestion Pipeline & Running the App
----------------------------------------------------------
A. Crawl Vibrant Wellness pages (hub + subpages) with crawl4ai
   - Seed file: include the category hub pages and allow the crawler to follow links to subpages.
   - Run:
        python crawl_vibrant.py
   - Outputs:
        data/raw_html/    (raw pages)
        data/md/          (markdown-converted pages)
        url_map.tsv       (id → url list)

B. Parse FAQs into structured Q/A
   - Run:
        python extract_qa.py
   - Outputs:
        qa.jsonl          (each line: {"kind":"qa","question","answer","url",...})
        kb.jsonl          (optional combined corpus if you created one)

C. (Optional) Build your own Neo4j KG from Q/A (already completed earlier)
   - Run (optional, if you want your custom KG in Neo4j):
        python build_kg.py
   - This step is NOT required for LightRAG retrieval but useful for exploration.

D. Ingest into LightRAG for retrieval (required for Phase 4’s “Retrieval with LightRAG”)
   - Run:
        python ingest_lightrag.py
   - What it does:
        • Indexes Q/A chunks (embeddings + keyword) in LightRAG’s storage
        • Extracts entities/relations and stores them in Neo4j (LightRAG’s own schema)
        • Enables “mix” retrieval (vector + keyword + KG with reranking)
   - Tips if it’s slow:
        • Ensure Ollama is using GPU (check with `nvidia-smi` while generating)
        • Lower num_ctx to 8192 in ingest_lightrag.py
        • Ingest a subset first (e.g., first 60 QAs), verify, then backfill

E. Run the FastAPI app
   - Start server:
        uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   - Open in browser:
        http://localhost:8000/
   - Usage:
        • Type your question, pick Top K if needed, and click "Ask"
        • Answer appears with inline bracket citations [1], [2]
        • Sources section lists the URLs matching those citations
     
Demo gif:

![demo](https://github.com/sjjgh/vibrant-rag/blob/main/demo.gif)

