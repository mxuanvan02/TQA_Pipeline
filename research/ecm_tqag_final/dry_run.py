from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecm_tqag.freeze import build_freeze, plan_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the offline ECM--TQAG freeze plan.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    freeze = build_freeze(args.manifest)
    ledger = plan_ledger(freeze)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "FREEZE_MANIFEST.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out / "DRY_RUN_LEDGER.jsonl").open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "DRY_RUN_READY",
        "freeze": str(args.out / "FREEZE_MANIFEST.json"),
        "ledger": str(args.out / "DRY_RUN_LEDGER.jsonl"),
        "rows": len(ledger),
        "construction_calls": sum(row["construction_calls"] for row in ledger),
        "http_calls": 0,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
