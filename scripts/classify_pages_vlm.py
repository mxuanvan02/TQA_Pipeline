#!/usr/bin/env python3
"""Classify rendered textbook pages as diagram vs text using a VLM via 9Router.
Reuses 9Router OpenAI-compatible chat/completions endpoint. Key from env only.
Output: JSONL with per-image label.
Usage: classify_pages_vlm.py <img_dir> <out_jsonl> [--limit N] [--glob PAT]
"""
from __future__ import annotations
import argparse, base64, json, os, sys, time, glob
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DEFAULT_MODEL = os.environ.get("NINEROUTER_VLM_MODEL", "cx/gpt-4o")
DEFAULT_ENDPOINT = "http://host.docker.internal:20128/v1"
# Duong dan .env: khai bao qua VLM_ENV_FILE, hoac do theo cac vi tri mac dinh.
# Khong hard-code duong dan home cua mot may cu the.
ENV_FILES = tuple(
    Path(p) for p in (
        os.environ.get("VLM_ENV_FILE"),
        Path.cwd() / ".env",
        Path("/opt/data/.env"),
        Path.home() / ".config/vlm/.env",
        Path.home() / "AI/hermes-stack/config/.env",
    ) if p
)

def _load_env():
    for p in ENV_FILES:
        if not p.exists(): continue
        for raw in p.read_text(errors="replace").splitlines():
            l=raw.strip()
            if not l or l.startswith("#") or "=" not in l: continue
            k,v=l.split("=",1); k=k.strip()
            if k and k not in os.environ: os.environ[k]=v.strip().strip('"').strip("'")

def _inside_docker(): return Path("/.dockerenv").exists()

def _endpoint():
    v=(os.environ.get("NINEROUTER_ENDPOINT") or os.environ.get("NINEROUTER_API_URL") or DEFAULT_ENDPOINT).strip().rstrip("/")
    if not _inside_docker(): v=v.replace("host.docker.internal","127.0.0.1")
    return v if v.endswith("/chat/completions") else f"{v}/chat/completions"

def _key():
    for k in ("NINEROUTER_API_KEY","NINEROUTER_KEY","OPENAI_API_KEY"):
        if os.environ.get(k): return os.environ[k]
    return ""

PROMPT = ("Bạn phân loại MỘT trang sách giáo trình luật. Trả lời DUY NHẤT một JSON: "
          '{"label":"<nhãn>","has_arrows":<true|false>,"note":"<ngắn>"}. '
          "Nhãn ∈ {flowchart_quy_trinh, so_do_to_chuc_cay, so_do_khac, bang_bieu, bieu_mau, van_ban_thuan, anh_khac}. "
          "flowchart_quy_trinh = có các hộp/bước nối bằng MŨI TÊN thể hiện trình tự/điều kiện. "
          "so_do_to_chuc_cay = sơ đồ phân cấp/cây cơ quan. bang_bieu = bảng kẻ ô. "
          "bieu_mau = mẫu đơn/tờ khai. van_ban_thuan = chỉ chữ. Chỉ trả JSON, không giải thích thêm.")

def classify(img_path, model):
    b64=base64.b64encode(Path(img_path).read_bytes()).decode()
    payload={"model":model,"max_tokens":150,"temperature":0,
        "messages":[{"role":"user","content":[
            {"type":"text","text":PROMPT},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}]}
    req=Request(_endpoint(), data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {_key()}"})
    last_err = None
    for attempt in range(2):
        try:
            with urlopen(req, timeout=120) as r:
                d=json.loads(r.read())
            txt=d["choices"][0]["message"]["content"].strip()
            txt=txt.replace("```json","").replace("```","").strip()
            try: obj=json.loads(txt)
            except: obj={"label":"PARSE_ERR","raw":txt[:120]}
            return obj
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(2 + attempt * 3)
    return {"label":"HTTP_ERR","note":str(last_err)[:120]}

def main():
    _load_env()
    ap=argparse.ArgumentParser()
    ap.add_argument("img_dir"); ap.add_argument("out_jsonl")
    ap.add_argument("--limit",type=int,default=0); ap.add_argument("--glob",default="*.jpg")
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--resume", action="store_true")
    a=ap.parse_args()
    files=sorted(glob.glob(os.path.join(a.img_dir,a.glob)))
    if a.limit: files=files[:a.limit]
    done=set()
    if a.resume and os.path.exists(a.out_jsonl):
        for line in open(a.out_jsonl,encoding="utf-8"):
            try:
                obj=json.loads(line); done.add(obj.get("file"))
            except Exception:
                pass
        files=[fp for fp in files if os.path.basename(fp) not in done]
    n=0
    mode="a" if a.resume else "w"
    with open(a.out_jsonl,mode,encoding="utf-8") as out:
        for fp in files:
            r=classify(fp,a.model); r["file"]=os.path.basename(fp)
            out.write(json.dumps(r,ensure_ascii=False)+"\n"); out.flush()
            n+=1
            print(f"[{n}/{len(files)}] {r.get('label'):24} {os.path.basename(fp)[:40]}",flush=True)
    print(f"DONE {n} new images -> {a.out_jsonl}; skipped {len(done)}",flush=True)

if __name__=="__main__": main()
