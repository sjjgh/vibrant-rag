# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 16:49:22 2025

@author: JIajie Shi
"""

import asyncio, os, hashlib, re, pathlib
from urllib.parse import urlparse, urljoin
from inspect import signature
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag



from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)

DATA_DIR = pathlib.Path("data")
RAW_DIR  = DATA_DIR / "raw_html"
MD_DIR   = DATA_DIR / "md"
RAW_DIR.mkdir(parents=True, exist_ok=True)
MD_DIR.mkdir(parents=True, exist_ok=True)
BASE_HOST = "vibrant-wellness.com"
#%%
# --- helpers ---



def canonicalize(u: str, base: str) -> str:
    # join relative, drop fragments, normalize host + path
    u = urljoin(base, u)
    u, _ = urldefrag(u)
    p = urlparse(u)
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    # collapse '//' and drop trailing slash (except root)
    path = re.sub(r"/{2,}", "/", p.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return p._replace(netloc=host, path=path, params="", query="", fragment="").geturl()

def is_allowed(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.netloc.endswith(BASE_HOST) and p.path.startswith("/tests")
    except Exception:
        return False

def make_run_cfg(**kwargs):
    """
    Build CrawlerRunConfig with only the kwargs supported by your installed crawl4ai.
    This avoids errors like 'unexpected keyword argument'.
    """
    try:
        params = signature(CrawlerRunConfig.__init__).parameters
        allowed = {k: v for k, v in kwargs.items() if k in params}
        return CrawlerRunConfig(**allowed)
    except Exception:
        # Last resort: no kwargs at all
        return CrawlerRunConfig()
    
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def same_domain(u: str, base_host: str) -> bool:
    try:
        return urlparse(u).netloc.endswith(base_host)
    except Exception:
        return False

def read_seeds(path="seeds.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

async def crawl_one(crawler, url: str):
    run_cfg = make_run_cfg(
        markdown_generator=DefaultMarkdownGenerator(),
        remove_overlay_elements=True,
        screenshot=False,
        respect_robots_txt=True,
        delay=1.0,
    )
    result = await crawler.arun(url=url, config=run_cfg)
    return result

async def crawl_site(seeds, max_pages=200, max_depth=2):
    # Determine base host from first seed
    base_host = urlparse(seeds[0]).netloc.split(":")[0]

    visited = set()
    queue = [(u, 0) for u in seeds]
    seen_urls = []

    browser_cfg = BrowserConfig(headless=True)
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        while queue and len(visited) < max_pages:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            if not same_domain(url, base_host):
                continue
            try:
                print(f"[crawl] depth={depth} {url}")
                res = await crawl_one(crawler, url)
                if not res or (not res.html and not res.markdown):
                    continue

                uid = sha1(url)
                (RAW_DIR / f"{uid}.html").write_text(res.html or "", encoding="utf-8")
                md_text = soft_clean_md(res.markdown or "")
                (MD_DIR  / f"{uid}.md").write_text(md_text, encoding="utf-8")
                seen_urls.append((uid, url))

                # enqueue same-domain links
                for link in res.links or []:
                    link = urljoin(url, link)
                    if same_domain(link, base_host) and link not in visited:
                        queue.append((link, depth + 1))

                visited.add(url)
            except Exception as e:
                print(f"[warn] {url} -> {e}")

    # save a small map for later citation
    with open(DATA_DIR / "url_map.tsv", "w", encoding="utf-8") as f:
        for uid, u in seen_urls:
            f.write(f"{uid}\t{u}\n")

def soft_clean_md(md: str) -> str:
    # strip repeated whitespace and very long nav lines
    import re
    md = re.sub(r'[ \t]+\n', '\n', md)
    # remove lines that look like nav crumbs or social bars
    drop_patterns = [
        r'(?i)breadcrumbs?', r'(?i)subscribe', r'(?i)share this', r'(?i)cookie',
        r'(?i)privacy policy', r'(?i)terms of use'
    ]
    lines = []
    for ln in md.splitlines():
        if any(re.search(p, ln) for p in drop_patterns):
            continue
        lines.append(ln)
    return '\n'.join(lines)
#%%
if __name__ == "__main__":
    seeds = read_seeds("seeds.txt")
    if not seeds:
        raise SystemExit("Put Vibrant seed URLs into seeds.txt (one per line).")
    asyncio.run(crawl_site(seeds))
