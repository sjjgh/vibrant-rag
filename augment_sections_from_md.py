# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 18:23:31 2025

@author: JIajie Shi
"""

import json, pathlib, re

DATA_DIR = pathlib.Path("data")
MD_DIR   = DATA_DIR / "md"
URL_MAP  = DATA_DIR / "url_map.tsv"
OUT      = pathlib.Path("sections.jsonl")

def norm(s:str)->str: return re.sub(r"\s+"," ",(s or "")).strip()

def load_url_map():
    m={}
    if URL_MAP.exists():
        for ln in URL_MAP.read_text(encoding="utf-8").splitlines():
            if "\t" in ln:
                uid,url=ln.split("\t",1); m[uid]=url
    return m

def extract_sections(md:str):
    lines = md.splitlines()
    headers=[]
    for i,ln in enumerate(lines):
        m=re.match(r"^(#{1,6})\s+(.*\S)\s*$", ln.strip())
        if m:
            level=len(m.group(1)); title=norm(m.group(2))
            headers.append((i,level,title))
    items=[]
    for idx,(i,lvl,title) in enumerate(headers):
        if lvl>4 or len(title)<3: continue
        end=len(lines)
        for j in range(idx+1,len(headers)):
            if headers[j][1] <= lvl:
                end = headers[j][0]; break
        body = norm("\n".join(lines[i+1:end]))
        if len(body)>=60 and len(title)<=180:
            items.append({"kind":"section","question":title,"answer":body})
    return items

def main():
    url_map=load_url_map()
    total=0
    with OUT.open("w",encoding="utf-8") as out:
        for p in sorted(MD_DIR.glob("*.md")):
            uid=p.stem; url=url_map.get(uid,"")
            md=p.read_text(encoding="utf-8",errors="ignore")
            sections=extract_sections(md)
            for it in sections:
                it.update({"url":url,"doc_id":uid,"section":None,"source_format":"md_section"})
                out.write(json.dumps(it,ensure_ascii=False)+"\n"); total+=1
    print(f"wrote {total} section items -> {OUT.resolve()}")

if __name__=="__main__":
    main()