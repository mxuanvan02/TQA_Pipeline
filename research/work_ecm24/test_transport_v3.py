#!/usr/bin/env python3
"""Regression test for the v3 TRANSPORT layer (parse + truncation detection).

Both bugs here were found on real pilot rows, not invented:

  * pilot_v3a/b: `is_truncated` only looked for an unclosed ``` fence, so a
    response whose `steps` array never closed (no fence at all) was misfiled as
    a contract rejection instead of being retried.
  * pilot_v3c: a response emitted a COMPLETE JSON object followed by a prose
    epilogue ("...}]}\n\nNote: Bước cuối không yêu cầu anchor..."). json.loads
    raised "Extra data" and the usable payload was thrown away.

Fixtures in audit/transport_fixtures_v3c.json are the verbatim raw strings from
runs/pilot_v3c. Negative controls assert the repairs do NOT rescue garbage:
a truly truncated array must still fail.

Scope: transport only. Nothing here touches a contract guard.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_ecm_v3 as r3  # noqa: E402

FAILED: list[str] = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


print("\n[1] real pilot_v3c fixtures")
fx = json.loads((ROOT / "audit/transport_fixtures_v3c.json").read_text(encoding="utf-8"))
check("two_fixtures_present", len(fx) == 2, f"n={len(fx)}")

# fixture 0: originally logged as "genuinely truncated" (unbalanced '[') and
# this test used to assert it stayed unparseable. That was WRONG: inspecting
# the full string shows 3 complete steps (hierarchy_traverse, last step
# `traverse` legitimately has no `anchor` because it is derived) with no
# mid-sentence cutoff anywhere -- it is missing exactly one `]` closing the
# `steps` array, the same shape as the 11 full_v3d fixtures in [7]. is_truncated
# still flags it (unbalanced brackets is a real signal), but parse_model_json's
# repair_missing_closers() now recovers it correctly instead of discarding a
# usable row. Kept as a regression test with the CORRECTED expectation.
p0, why0 = r3.parse_model_json(fx[0])
check("truncated_fixture_flagged_by_is_truncated", r3.is_truncated(fx[0]))
check("truncated_fixture_recovered_via_bracket_repair",
      isinstance(p0, dict) and isinstance(p0.get("steps"), list) and len(p0["steps"]) == 3,
      str(why0) if p0 is None else str(p0.get("motif")))

# fixture 1: complete JSON + prose epilogue -> must be recovered
p1, why1 = r3.parse_model_json(fx[1])
check("prose_epilogue_not_flagged_truncated", not r3.is_truncated(fx[1]))
check("prose_epilogue_recovered", isinstance(p1, dict), str(why1))
check("recovered_payload_is_usable",
      isinstance(p1, dict) and p1.get("motif") == "hierarchy_traverse"
      and isinstance(p1.get("steps"), list) and len(p1["steps"]) == 3,
      str(p1.get("motif") if isinstance(p1, dict) else None))

print("\n[2] is_truncated: positives")
check("unclosed_fence", r3.is_truncated('```json\n{"a":1}'))
check("unclosed_array", r3.is_truncated('{"steps":[{"op":"locate"}'))
check("unclosed_string", r3.is_truncated('{"q":"chua dong'))

print("\n[3] is_truncated: negatives (must NOT fire)")
check("plain_object", not r3.is_truncated('{"a":1}'))
check("closed_fence", not r3.is_truncated('```json\n{"a":1}\n```'))
check("braces_inside_string", not r3.is_truncated('{"a":"co dau { va [ trong cau"}'))
check("object_plus_prose", not r3.is_truncated('{"a":1}\n\nNote: xong.'))

print("\n[4] extract_first_json_object")
check("slices_trailing_prose",
      r3.extract_first_json_object('{"a":1}\n\nNote: x') == '{"a":1}')
check("ignores_braces_in_string",
      r3.extract_first_json_object('{"a":"x } y"}') == '{"a":"x } y"}')
check("returns_none_on_truncated",
      r3.extract_first_json_object('{"a":[1,2') is None)
check("returns_none_on_prose_only",
      r3.extract_first_json_object("khong co json o day") is None)

print("\n[5] parse_model_json: negative controls (no rescuing garbage)")
for name, raw in [("truncated_array", '{"steps":[{"op":"locate"}'),
                  ("prose_only", "khong co json"),
                  ("empty", ""),
                  ("json_array_not_object", "[1,2,3]")]:
    parsed, why = r3.parse_model_json(raw)
    check(f"reject::{name}", parsed is None, str(why))

print("\n[6] parse_model_json: accept paths")
for name, raw in [("plain", '{"a":1}'),
                  ("fenced", '```json\n{"a":1}\n```'),
                  ("plus_prose", '{"a":1}\n\nNote: xong.'),
                  ("brace_in_string", '{"a":"x { y"}')]:
    parsed, why = r3.parse_model_json(raw)
    check(f"accept::{name}", isinstance(parsed, dict), str(why))

print("\n[7] repair_missing_closers: real full_v3d fixtures (missing ']')")
# full_v3d found 11/48 rows where the model wrote a 3-step `steps` array whose
# LAST step is derived (no `anchor` key) and closed with `}}` instead of
# `]}}` -- the array-closer for `steps` was dropped. All 11 are verbatim raw
# stage1 strings, not invented. An earlier version of this repair naively
# appended closers in LIFO order at the END of the string, which produced the
# WRONG structure here (mismatch happens mid-string, not at the tail) and
# recovered 0/11. The fix inserts the missing bracket AT the mismatch point.
fx11 = json.loads((ROOT / "audit/missing_bracket_fixtures_v3d.json").read_text(encoding="utf-8"))
check("eleven_fixtures_present", len(fx11) == 11, f"n={len(fx11)}")
n_recovered = 0
for i, raw in enumerate(fx11):
    parsed, why = r3.parse_model_json(raw)
    if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list):
        n_recovered += 1
    else:
        print(f"    fixture[{i}] NOT recovered :: {why}")
check("all_eleven_recovered", n_recovered == len(fx11), f"{n_recovered}/{len(fx11)}")

print("\n[8] repair_missing_closers: negative controls (no rescuing garbage)")
check("still_rejects_truncated_array",
      r3.repair_missing_closers('{"steps":[{"op":"locate"}') is None)
check("still_rejects_prose_only",
      r3.repair_missing_closers("khong co json o day") is None)
check("still_rejects_empty", r3.repair_missing_closers("") is None)
check("still_rejects_unterminated_string",
      r3.repair_missing_closers('{"q":"chua dong') is None)
check("valid_plain_untouched", r3.repair_missing_closers('{"a":1}') is None)
check("valid_nested_untouched",
      r3.repair_missing_closers('{"a":[{"x":1},{"y":2}]}') is None)
# a real double-truncation (two closers missing, nothing after) must NOT be
# silently patched by this repair -- that belongs to is_truncated's territory.
check("double_missing_not_silently_fixed",
      r3.repair_missing_closers('{"a":[{"x":1},{"y":2}') is None)

total = 0
with open(__file__, encoding="utf-8") as fh:
    total = sum(1 for line in fh if line.startswith("check(") or line.strip().startswith("check("))

print("\n" + "=" * 60)
n_fail = len(FAILED)
print(f"TRANSPORT SELF-TEST v3: {'ALL PASSED' if not n_fail else f'{n_fail} FAILED: {FAILED}'}")
print("Scope: transport repair only; contract guards untouched.")
sys.exit(1 if n_fail else 0)
