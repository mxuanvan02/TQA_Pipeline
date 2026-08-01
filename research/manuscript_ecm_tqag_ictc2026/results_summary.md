# ECM-TQAG exploratory result summary

## Scope

- Model: `qwen/qwen3.7-plus` via OpenRouter
- Matrix: 8 chunks × 3 evidence conditions × 3 methods = 72 cells
- Final parsed: 60/72 (83.3%)
- Final rejected: 12/72 (16.7%)
- Rerun: 18 cells from two chunks after context augmentation

## Provenance partitions

| Partition | Records | Parsed | Rejected | Evidence manifest |
|---|---:|---:|---:|---|
| Unaffected/original | 54 | 47 | 7 | `ecm_inputs_8chunks_v3.json` |
| Rerun/augmented | 18 | 13 | 5 | `ecm_inputs_8chunks_v3_context_augmented.json` |
| Total | 72 | 60 | 12 | mixed, audited by partition |

## Rejection taxonomy

| Scenario | Count | Cells |
|---|---:|---|
| Insufficient evidence | 5 | 4 on `LUAT CHUNG KHOAN_chunk_0`; 1 on `47. LY LUAN DINH TOI DANH_chunk_40` |
| Non-literal source grounding | 4 | 3 on `47. LY LUAN DINH TOI DANH_chunk_40`; 1 on `19.LUAT HIEN PHAP NUOC NGOAI_chunk_125` |
| ECM answer not bound to executor atom | 2 | `19.LUAT HIEN PHAP NUOC NGOAI_chunk_44`; `17,18. LUAT HIEN PHAP VIET NAM_chunk_79` |
| Invalid ECM evidence anchor | 1 | `47. LY LUAN DINH TOI DANH_chunk_50` |

The insufficiency cases are evidence-coverage outcomes. The other cases are model-output/contract failures detected by deterministic replay. The two short packages were not padded with unrelated neighbouring text: the French-court package was augmented with its semantically adjacent preceding section, while the securities package remained title-plus-image because no safe adjacent OCR context described that image.

## Release boundary

The summary is safe for repository documentation. Raw JSONL ledgers, textbook text, and page images remain local and are not uploaded without rights review. A parsed item means only that the structural/provenance contract passed; it does not establish legal correctness, one-best-answer validity, pedagogical quality, or genuine visual necessity.