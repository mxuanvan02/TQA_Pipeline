#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os, re, time
from pathlib import Path
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
from urllib.request import Request, urlopen

PDF_RE = re.compile(r'https?://[^\s"<>]+?\.pdf(?:\?[^\s"<>]*)?', re.I)
DDG_RE = re.compile(r'href="(?P<href>//duckduckgo\.com/l/\?[^"#]+)"')


def fetch_text(url: str, timeout: int = 30) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def resolve_ddg(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    qs = parse_qs(urlparse(href).query)
    return unquote(qs.get("uddg", [href])[0])


def search_ddg(query: str) -> list[str]:
    html = fetch_text("https://duckduckgo.com/html/?q=" + quote_plus(query))
    urls = set(PDF_RE.findall(html))
    for m in DDG_RE.finditer(html):
        u = resolve_ddg(m.group("href"))
        if ".pdf" in u.lower():
            urls.add(u)
    return sorted(urls)


def safe_name(url: str) -> str:
    name = unquote(Path(urlparse(url).path).name) or "download.pdf"
    name = re.sub(r'[^A-Za-z0-9._ -]+', '_', name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:180]


def download(url: str, out: Path) -> tuple[bool, str]:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/pdf,*/*"})
        with urlopen(req, timeout=60) as r:
            ctype = r.headers.get("content-type", "")
            data = r.read()
        if len(data) < 50_000:
            return False, f"too_small:{len(data)} content_type:{ctype}"
        if not (data[:4] == b"%PDF" or "pdf" in ctype.lower()):
            return False, f"not_pdf content_type:{ctype} size:{len(data)}"
        out.write_bytes(data)
        return True, f"ok:{len(data)}"
    except Exception as e:
        return False, f"{type(e).__name__}:{str(e)[:160]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--failed", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seen_urls: set[str] = set()
    rows = []
    failed = []
    for q in args.queries:
        print("QUERY", q, flush=True)
        try:
            urls = search_ddg(q)
        except Exception as e:
            failed.append(("", q, f"search:{type(e).__name__}:{e}")); continue
        for url in urls:
            if url in seen_urls or len(rows) >= args.limit:
                continue
            seen_urls.add(url)
            domain = urlparse(url).netloc.lower()
            name = safe_name(url)
            out = out_dir / name
            if out.exists() and out.stat().st_size > 50_000:
                ok, note = True, f"exists:{out.stat().st_size}"
            else:
                ok, note = download(url, out)
                time.sleep(1)
            if ok:
                rows.append({"filename": out.name, "title": out.stem, "url": url, "source_domain": domain, "source_type": "external_public_pdf", "notes": note})
                print("DOWNLOADED", out, note, flush=True)
            else:
                failed.append((url, q, note))
                print("FAILED", url, note, flush=True)
        if len(rows) >= args.limit:
            break

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "title", "url", "source_domain", "source_type", "notes"])
        w.writeheader(); w.writerows(rows)
    with open(args.failed, "w", encoding="utf-8") as f:
        for url, q, note in failed:
            f.write(f"{url}\t{q}\t{note}\n")
    print(f"DONE downloaded={len(rows)} failed={len(failed)} out={out_dir}", flush=True)

if __name__ == "__main__":
    main()
