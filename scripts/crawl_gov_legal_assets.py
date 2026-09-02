#!/usr/bin/env python3
"""Crawl public Vietnamese legal-education/government pages for PDFs and images.

This is a conservative collector: it stores raw assets plus provenance metadata.
It does not infer legal labels or redistribute claims; downstream review handles that.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html.parser
import os
import re
import ssl
import time
import zlib
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

ASSET_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
PAGE_EXTS = {"", ".html", ".htm", ".aspx", ".php"}
KEYWORDS = [
    "phap-luat", "pháp-luật", "phapluat", "luat", "luật", "pbgdpl",
    "pho-bien", "phổ-biến", "giao-duc", "giáo-dục", "thu-tuc", "thủ-tục",
    "quy-trinh", "quy-trình", "huong-dan", "hướng-dẫn", "hoi-dap", "hỏi-đáp",
    "infographic", "do-hoa", "đồ-họa", "so-do", "sơ-đồ", "bieu-do", "biểu-đồ",
]


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for k, v in attrs:
            if k.lower() in {"href", "src"} and v:
                self.links.append(v)


def norm_url(url: str) -> str:
    return urldefrag(url)[0].strip()


def host_ok(seed_host: str, url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == seed_host or host.endswith("." + seed_host)


def has_keyword(url: str) -> bool:
    low = url.lower()
    return any(k in low for k in KEYWORDS)


def safe_name(url: str, content: bytes) -> str:
    parsed = urlparse(url)
    base = Path(parsed.path).name or "asset"
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base)[:140]
    ext = Path(parsed.path).suffix.lower()
    if ext not in ASSET_EXTS:
        ext = ".bin"
    digest = hashlib.sha256(content).hexdigest()[:12]
    stem = base[:-len(ext)] if base.lower().endswith(ext) else base
    return f"{stem}_{digest}{ext}"


def fetch(url: str, timeout: int) -> tuple[bytes, str]:
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/pdf,image/*,*/*",
        "Accept-Encoding": "gzip, deflate",
    })
    context = ssl._create_unverified_context() if url.lower().startswith("https://toaan.gov.vn") else None
    with urlopen(req, timeout=timeout, context=context) as r:
        data = r.read()
        enc = (r.headers.get("content-encoding") or "").lower()
        # Some Vietnamese government portals gzip HTML without urllib auto-decoding it.
        if enc == "gzip" or data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        elif enc == "deflate":
            data = zlib.decompress(data)
        return data, r.headers.get("content-type", "")


def classify_url(url: str, content_type: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext == ".pdf" or "pdf" in content_type.lower():
        return "pdf"
    if ext in {".png", ".jpg", ".jpeg", ".webp"} or "image/" in content_type.lower():
        return "image"
    return "page"


def crawl(seed: str, out_root: Path, max_pages: int, max_assets: int, timeout: int, delay: float) -> None:
    seed = norm_url(seed)
    seed_host = urlparse(seed).netloc.lower()
    domain_dir = out_root / re.sub(r"[^A-Za-z0-9.-]+", "_", seed_host)
    pdf_dir = domain_dir / "pdf"
    img_dir = domain_dir / "images"
    meta_dir = domain_dir / "metadata"
    for d in (pdf_dir, img_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    q: deque[str] = deque([seed])
    pages = 0
    assets = 0
    rows: list[dict[str, str]] = []
    failures: list[tuple[str, str]] = []

    while q and assets < max_assets:
        url = norm_url(q.popleft())
        ext0 = Path(urlparse(url).path).suffix.lower()
        if url in seen or (not host_ok(seed_host, url) and ext0 not in ASSET_EXTS):
            continue
        if ext0 not in ASSET_EXTS and pages >= max_pages:
            continue
        seen.add(url)
        try:
            data, ctype = fetch(url, timeout)
        except Exception as e:
            failures.append((url, f"fetch:{type(e).__name__}:{str(e)[:160]}"))
            continue

        kind = classify_url(url, ctype)
        ext = Path(urlparse(url).path).suffix.lower()
        if kind in {"pdf", "image"}:
            if len(data) < 8_000:
                failures.append((url, f"too_small:{len(data)}:{ctype}"))
            else:
                target_dir = pdf_dir if kind == "pdf" else img_dir
                fn = safe_name(url, data)
                out = target_dir / fn
                out.write_bytes(data)
                rows.append({
                    "filename": str(out.relative_to(out_root)),
                    "asset_type": kind,
                    "url": url,
                    "source_domain": seed_host,
                    "content_type": ctype,
                    "bytes": str(len(data)),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "notes": "public_crawl_candidate",
                })
                assets += 1
                print("ASSET", kind, out, len(data), flush=True)
        elif ext in PAGE_EXTS:
            pages += 1
            text = data.decode("utf-8", "replace")
            p = LinkParser(); p.feed(text)
            links = list(p.links)
            # Some portals embed downloadable CDN URLs in escaped JSON rather than href/src.
            links.extend(re.findall(r"https?:\\?/\\?/[^\\\"'<>\s]+", text))
            for link in links:
                link = link.replace("\\/", "/")
                u = norm_url(urljoin(url, link))
                if u in seen:
                    continue
                u_ext = Path(urlparse(u).path).suffix.lower()
                if not host_ok(seed_host, u) and u_ext not in ASSET_EXTS:
                    continue
                # Allow CDN-hosted PDFs/images referenced by a trusted page; keep page crawling same-domain.
                if u_ext in ASSET_EXTS or has_keyword(u) or has_keyword(url) or (u_ext in PAGE_EXTS and pages <= 8):
                    q.append(u)
            print("PAGE", pages, url, "queue", len(q), "assets", assets, flush=True)
        time.sleep(delay)

    manifest = meta_dir / "assets_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "asset_type", "url", "source_domain", "content_type", "bytes", "sha256", "retrieved_at", "notes"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    (meta_dir / "failed.txt").write_text("\n".join(f"{u}\t{e}" for u, e in failures), encoding="utf-8")
    print(f"DONE seed={seed} pages={pages} assets={assets} failures={len(failures)} manifest={manifest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out-root", default="data/raw_gov")
    ap.add_argument("--max-pages", type=int, default=120)
    ap.add_argument("--max-assets", type=int, default=80)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()
    crawl(args.seed, Path(args.out_root), args.max_pages, args.max_assets, args.timeout, args.delay)


if __name__ == "__main__":
    main()
