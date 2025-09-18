# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 18:24:41 2025

@author: JIajie Shi
"""

import json, hashlib, pathlib, re

FAQ = pathlib.Path("qa.jsonl")
SECT = pathlib.Path("sections.jsonl")
OUT = pathlib.Path("kb.jsonl")

def sha1(s:str)->str: return hashlib.sha1(s.encode("utf-8")).hexdigest()

def norm(s:str)->str: return re.sub(r"\s+"," ",(s or "")).strip()

def load_jsonl(p: pathlib.Path):
    if not p.exists(): return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    items = load_jsonl(FAQ) + load_jsonl(SECT)
    seen = set()
    cleaned = []
    for it in items:
        q = norm(it.get("question","")); a = norm(it.get("answer",""))
        url = it.get("url",""); kind = it.get("kind","section")
        if not q or not a: continue
        # exact-dupe guard
        key = (q.lower(), a.lower())
        if key in seen: continue
        seen.add(key)
        # clip overlong answers (keeps prompts tight)
        if len(a) > 2000: a = a[:2000] + " ..."
        _id = sha1(f"{url}::{q}")
        cleaned.append({
            "id": _id,
            "kind": kind,
            "question": q,
            "answer": a,
            "url": url,
            "doc_id": it.get("doc_id"),
            "section": it.get("section"),
            "source_format": it.get("source_format")
        })
    with OUT.open("w",encoding="utf-8") as f:
        for it in cleaned:
            f.write(json.dumps(it,ensure_ascii=False)+"\n")
    print(f"final kb: {len(cleaned)} items -> {OUT.resolve()}")

if __name__=="__main__":
    main()
