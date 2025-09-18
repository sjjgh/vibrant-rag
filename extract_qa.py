# extract_faq_md.py
# Parse FAQ sections from Markdown files under data/md and save QA pairs to qa.jsonl
import json, pathlib, re

DATA_DIR = pathlib.Path("data")
MD_DIR   = DATA_DIR / "md"
URL_MAP_PATH = DATA_DIR / "url_map.tsv"
OUT_PATH = pathlib.Path("qa.jsonl")

# ---- utils ----
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def load_url_map(path: pathlib.Path):
    m = {}
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            if "\t" in ln:
                uid, url = ln.split("\t", 1)
                m[uid] = url
    return m

# Match headings: #, ##, ###, ####, etc.
HDR_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Lines to ignore inside FAQ sections (e.g., "View all FAQs" links)
IGNORE_LINE_RE = re.compile(r"^\s*\[\s*view\s+all\s+faqs?\s*\]\([^)]+\)\s*$", re.IGNORECASE)

def find_headers(lines):
    """Return list of (idx, level, title) for all markdown headers."""
    headers = []
    for i, ln in enumerate(lines):
        m = HDR_RE.match(ln.strip())
        if m:
            level = len(m.group(1))
            title = norm(m.group(2))
            headers.append((i, level, title))
    return headers

def faq_sections(lines):
    """
    Yield (start_idx, end_idx, section_title) for each FAQ section:
    header level <= 3 containing 'faq' (FAQ/FAQs/Frequently Asked Questions).
    """
    headers = find_headers(lines)
    for h_idx, (i, lvl, title) in enumerate(headers):
        if lvl <= 3 and re.search(r"\bf(?:requently\s+asked\s+questions|aqs?)\b", title, re.IGNORECASE):
            # section ends at next header of same or higher importance (<= lvl)
            end = len(lines)
            for j in range(h_idx + 1, len(headers)):
                _i2, lvl2, _t2 = headers[j]
                if lvl2 <= lvl:
                    end = _i2
                    break
            yield i, end, title

def extract_qas_from_faq_block(block_lines):
    """
    Inside an FAQ section, treat ###/####/##### lines that look like questions
    as Q, and capture subsequent non-header lines as A until next Q/header.
    """
    items = []
    i = 0
    while i < len(block_lines):
        ln = block_lines[i].rstrip()
        # Skip empty / ignored utility lines
        if not ln.strip() or IGNORE_LINE_RE.match(ln):
            i += 1
            continue

        m = HDR_RE.match(ln.strip())
        if m:
            lvl = len(m.group(1))
            title = norm(m.group(2))
            # Consider sub-headers (### or deeper) as potential questions
            if lvl >= 3 and 5 <= len(title) <= 200 and (title.endswith("?") or re.search(r"\b(what|how|why|who|when|where|should|do)\b", title, re.I)):
                q = title
                i += 1
                ans_lines = []
                while i < len(block_lines):
                    ln2 = block_lines[i].rstrip()
                    if not ln2.strip():
                        # allow single blank line inside answer
                        ans_lines.append("")
                        i += 1
                        continue
                    m2 = HDR_RE.match(ln2.strip())
                    if m2 and len(m2.group(1)) >= 3:
                        # next sub-header => end of this answer
                        break
                    if IGNORE_LINE_RE.match(ln2):
                        i += 1
                        continue
                    ans_lines.append(ln2)
                    i += 1
                a = norm("\n".join(ans_lines))
                if len(a) >= 20:  # avoid trivial answers
                    items.append((q, a))
                continue
        # If not a header, just advance
        i += 1
    return items

def main():
    url_map = load_url_map(URL_MAP_PATH)
    seen = set()
    total = 0

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for md_path in sorted(MD_DIR.glob("*.md")):
            doc_id = md_path.stem
            url = url_map.get(doc_id, "")
            text = md_path.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()

            found_any = False
            for start, end, sec_title in faq_sections(lines):
                block = lines[start:end]
                qas = extract_qas_from_faq_block(block)
                for q, a in qas:
                    key = (q, a)
                    if key in seen:
                        continue
                    seen.add(key)
                    item = {
                        "kind": "qa",
                        "question": q,
                        "answer": a,
                        "url": url,
                        "doc_id": doc_id,
                        "section": sec_title,           # e.g., "FAQs for Food Sensitivity Profile 2"
                        "source_format": "md_faq"
                    }
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total += 1
                    found_any = True

            # (Optional) If you want to log which files had FAQs:
            # if found_any:
            #     print(f"[ok] FAQs found in {md_path.name}")

    print(f"wrote {total} FAQ Q/A items to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
