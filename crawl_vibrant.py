# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 17:45:46 2025

@author: JIajie Shi
"""

# crawl_vibrant.py
import asyncio
import hashlib
import pathlib
import re
from inspect import signature
from urllib.parse import urlparse, urljoin, urldefrag

from bs4 import BeautifulSoup
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
)

# ---------- Paths ----------
DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw_html"
MD_DIR = DATA_DIR / "md"
RAW_DIR.mkdir(parents=True, exist_ok=True)
MD_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def read_seeds(path="seeds.txt"):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]

def soft_clean_md(md: str) -> str:
    """Light cleanup for obvious boilerplate lines; keep this conservative."""
    if not md:
        return ""
    md = re.sub(r"[ \t]+\n", "\n", md)
    drop_patterns = [
        r"(?i)breadcrumbs?", r"(?i)subscribe", r"(?i)cookie",
        r"(?i)privacy policy", r"(?i)terms of use", r"(?i)share this",
    ]
    out = []
    for ln in md.splitlines():
        if any(re.search(p, ln) for p in drop_patterns):
            continue
        out.append(ln)
    return "\n".join(out)

def base_host_from_url(u: str) -> str:
    p = urlparse(u)
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host

def canonicalize(u: str, base: str) -> str:
    """Join relative -> absolute, drop fragments/query, normalize host/path."""
    try:
        u = urljoin(base, u)
        u, _ = urldefrag(u)
        p = urlparse(u)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        # collapse '//' and trim trailing slash (except '/')
        path = re.sub(r"/{2,}", "/", p.path or "/")
        if path.endswith("/") and len(path) > 1:
            path = path[:-1]
        return p._replace(netloc=host, path=path, params="", query="", fragment="").geturl()
    except Exception:
        return u

def is_allowed(u: str, base_host: str) -> bool:
    try:
        p = urlparse(u)
        return p.netloc.endswith(base_host) and p.path.startswith("/tests")
    except Exception:
        return False

def make_run_cfg(**kwargs):
    """
    Build CrawlerRunConfig with only kwargs supported by your installed crawl4ai.
    This avoids 'unexpected keyword argument' errors on older versions (e.g., 0.7.4).
    """
    try:
        params = signature(CrawlerRunConfig.__init__).parameters
        allowed = {k: v for k, v in kwargs.items() if k in params}
        return CrawlerRunConfig(**allowed)
    except Exception:
        return CrawlerRunConfig()

# ---------- Crawl ----------
async def crawl_one(crawler: AsyncWebCrawler, url: str):
    # Keep the config minimal for wide version compatibility
    run_cfg = make_run_cfg(
        markdown_generator=DefaultMarkdownGenerator(),
        remove_overlay_elements=True,
        screenshot=False,
        respect_robots_txt=True,
        delay=1.0,
    )
    res = await crawler.arun(url=url, config=run_cfg)
    return res  # has .html, .markdown, .links

async def crawl_site(seeds, max_pages=600, max_depth=3):
    if not seeds:
        raise SystemExit("Put Vibrant seed URLs into seeds.txt (one per line).")

    base_host = base_host_from_url(seeds[0])
    visited = set()
    enqueued = set()
    queue = []

    # prime the queue with canonicalized seeds
    for s in seeds:
        cu = canonicalize(s, base=s)
        queue.append((cu, 0))
        enqueued.add(cu)

    browser_cfg = BrowserConfig(headless=True)
    saved = []

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        print("[INIT].... → Crawl4AI ready")
        while queue and len(visited) < max_pages:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            if not is_allowed(url, base_host):
                continue

            try:
                print(f"[crawl] depth={depth} {url}")
                res = await crawl_one(crawler, url)
                got = bool(res and (res.markdown or res.html))
                num_links = len(res.links or []) if res else 0
                print(f"  -> got: {got}  links(listed): {num_links}")

                if not res:
                    visited.add(url)
                    continue

                uid = sha1(url)
                # save HTML
                (RAW_DIR / f"{uid}.html").write_text(res.html or "", encoding="utf-8")
                # save cleaned Markdown
                md_text = soft_clean_md(res.markdown or "")
                (MD_DIR / f"{uid}.md").write_text(md_text, encoding="utf-8")
                saved.append((uid, url))

                # discover links from both res.links and HTML anchors
                links = set(res.links or [])
                if res.html:
                    soup = BeautifulSoup(res.html, "lxml")
                    for a in soup.find_all("a", href=True):
                        links.add(a["href"])

                for lnk in links:
                    lnk = canonicalize(lnk, base=url)
                    if is_allowed(lnk, base_host) and (lnk not in visited) and (lnk not in enqueued):
                        queue.append((lnk, depth + 1))
                        enqueued.add(lnk)

                visited.add(url)

            except Exception as e:
                print(f"[warn] {url} -> {e}")

    # write URL map for citations
    URL_MAP = DATA_DIR / "url_map.tsv"
    with URL_MAP.open("w", encoding="utf-8") as f:
        for uid, u in saved:
            f.write(f"{uid}\t{u}\n")
    print(f"[done] Saved {len(saved)} pages")
    print(f"[done] Wrote URL map -> {URL_MAP.resolve()}")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    seeds = read_seeds("seeds.txt")
    # Make it safe in Spyder/Jupyter too
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False

    if running:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(crawl_site(seeds))
    else:
        asyncio.run(crawl_site(seeds))
