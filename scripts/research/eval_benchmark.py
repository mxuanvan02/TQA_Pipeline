#!/usr/bin/env python3
"""
scripts/research/eval_benchmark.py — Zero-shot MCQ Benchmark Evaluation.

This script evaluates an LLM's zero-shot performance on the DHH-LegalQA dataset.
It supports:
- Accuracy@1 (Overall)
- Breakdown by Bloom's Taxonomy level (Remember, Understand, Apply)
- Breakdown by Modality (Text-only vs. Multimodal)
- Automated generation of TeX macros for Section 6 of the paper.

Inference: Powered by vLLM (high-throughput PagedAttention).
Dataset Schema: TQARecord JSONL.
"""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mcq_utils import resolve_ground_truth_index


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _get_choice_letter(index: int) -> str:
    return chr(65 + index)  # 0->A, 1->B, 2->C, 3->D


def _extract_answer_letter(text: str) -> str:
    """Extract the choice letter (A, B, C, D) from LLM output."""
    t = text.strip().upper()
    # Strategy 1: Look for exactly one of the letters at the start
    match = re.search(r"^[ ]*([A-D])[\s:.)]", t)
    if match:
        return match.group(1)
    
    # Strategy 2: Look for strings like "The correct answer is A"
    match = re.search(r"CORRECT ANSWER IS[ ]*([A-D])", t)
    if match:
        return match.group(1)

    # Strategy 3: Just find the first A, B, C, or D if the output is short
    if len(t) < 50:
        match = re.search(r"([A-D])", t)
        if match:
            return match.group(1)
    
    return "N/A"


def _infer_quantization(model_id: str) -> str | None:
    model_id_lower = model_id.lower()
    if "awq" in model_id_lower:
        return "awq"
    return None


def _truncate_context(text: str, max_chars: int | None) -> str:
    cleaned = " ".join(str(text).split())
    if max_chars is None or max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    if max_chars < 400:
        return cleaned[:max_chars].rstrip()

    head = int(max_chars * 0.7)
    tail = max_chars - head - len("\n...\n")
    if tail <= 0:
        return cleaned[:max_chars].rstrip()
    return f"{cleaned[:head].rstrip()}\n...\n{cleaned[-tail:].lstrip()}"


def build_mcq_prompt(record: dict[str, Any], max_context_chars: int | None = None) -> str:
    """
    Construct a zero-shot MCQ prompt for the LLM.
    Uses the context, question, and candidate answers.
    """
    ctx = _truncate_context(record.get("context_payload", {}).get("text", ""), max_context_chars)
    q = record.get("question_content", "")
    options = record.get("candidate_answers", [])
    
    options_str = "\n".join([f"{_get_choice_letter(i)}. {opt}" for i, opt in enumerate(options)])
    
    prompt = f"""\
Dựa trên ngữ cảnh pháp lý sau đây, hãy trả lời câu hỏi trắc nghiệm bên dưới.
Chỉ trả lời bằng CHỮ CÁI (A, B, C, hoặc D) đại diện cho đáp án đúng nhất.

### Ngữ cảnh:
{ctx}

### Câu hỏi:
{q}

### Các lựa chọn:
{options_str}

### Đáp án đúng (Chỉ chọn A, B, C hoặc D):"""
    return prompt


def build_mcq_prompt_no_context(record: dict[str, Any]) -> str:
    q = record.get("question_content", "")
    options = record.get("candidate_answers", [])
    options_str = "\n".join([f"{_get_choice_letter(i)}. {opt}" for i, opt in enumerate(options)])

    return f"""\
Hãy trả lời câu hỏi trắc nghiệm pháp lý sau đây.
Chỉ trả lời bằng CHỮ CÁI (A, B, C, hoặc D) đại diện cho đáp án đúng nhất.

### Câu hỏi:
{q}

### Các lựa chọn:
{options_str}

### Đáp án đúng (Chỉ chọn A, B, C hoặc D):"""


def _load_evaluable_records(
    dataset_path: Path,
    limit: int | None = None,
    split: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    raw_records = _load_jsonl(dataset_path)
    if split:
        raw_records = [r for r in raw_records if str(r.get("split", "")).strip() == split]
    if limit:
        raw_records = raw_records[:limit]

    records: list[dict[str, Any]] = []
    skip_reasons = Counter()
    for record in raw_records:
        if isinstance(record.get("gold_index"), int):
            gold_index = int(record["gold_index"])
            candidate_answers = record.get("candidate_answers", [])
            if isinstance(candidate_answers, list) and len(candidate_answers) == 4 and 0 <= gold_index <= 3:
                record = dict(record)
                record["_gold_index"] = gold_index
                records.append(record)
                continue

        resolution = resolve_ground_truth_index(record.get("candidate_answers", []), str(record.get("ground_truth", "")))
        matched_index = resolution["matched_index"]
        if resolution["candidate_count"] != 4:
            skip_reasons["not_4_options"] += 1
            continue
        if matched_index is None or matched_index < 0 or matched_index > 3:
            skip_reasons[resolution["method"]] += 1
            continue
        record = dict(record)
        record["_gold_index"] = matched_index
        records.append(record)

    return raw_records, records, skip_reasons


def _init_llm(
    model_id: str,
    gpu_util: float,
    tensor_parallel: int,
    max_model_len: int,
    max_num_seqs: int,
):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[ERROR] vLLM not found. Please pip install vllm.")
        return None, None

    quantization = _infer_quantization(model_id)
    llm_kwargs = {
        "model": model_id,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_util,
        "tensor_parallel_size": tensor_parallel,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "enforce_eager": False,
        "disable_log_stats": True,
    }
    if quantization:
        llm_kwargs["quantization"] = quantization

    from vllm.sampling_params import GuidedDecodingParams
    
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        max_tokens=4,
        temperature=0.0,
        guided_decoding=GuidedDecodingParams(choice=["A", "B", "C", "D"])
    )
    return llm, sampling_params


def _init_api_client(api_base: str, api_key: str):
    try:
        from openai import OpenAI
        return OpenAI(base_url=api_base, api_key=api_key)
    except ImportError:
        print("[ERROR] openai not found. Please pip install openai.")
        return None


class APIModelWrapper:
    """Mock vLLM interface for OpenAI-compatible APIs."""
    def __init__(self, client, model_id: str, max_tokens: int = 4, temperature: float = 0.0):
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        prompts: list[str],
        sampling_params=None,
        use_tqdm: bool = False,
        progress_label: str | None = None,
        progress_log_path: Path | None = None,
    ) -> list[Any]:
        def _call_api(prompt: str):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                # Create a mock vLLM output object
                class MockOutput:
                    def __init__(self, text):
                        self.outputs = [type('obj', (object,), {'text': text})]
                return MockOutput(response.choices[0].message.content)
            except Exception as e:
                print(f"[ERROR] API call failed: {e}")
                class ErrorOutput:
                    def __init__(self):
                        self.outputs = [type('obj', (object,), {'text': ""})]
                return ErrorOutput()

        max_workers = 8
        results: list[Any] = [None] * len(prompts)
        start_time = time.perf_counter()
        completed = 0

        def _write_progress() -> None:
            if not progress_log_path:
                return
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            rate = completed / elapsed
            remaining = len(prompts) - completed
            eta_sec = (remaining / rate) if rate > 0 else 0.0
            payload = {
                "label": progress_label or "api_inference",
                "completed": completed,
                "total": len(prompts),
                "progress_ratio": round(completed / len(prompts), 6) if prompts else 0.0,
                "elapsed_sec": round(elapsed, 2),
                "rate_req_per_sec": round(rate, 4),
                "eta_sec": round(eta_sec, 2),
            }
            progress_log_path.parent.mkdir(parents=True, exist_ok=True)
            progress_log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_call_api, prompt): idx for idx, prompt in enumerate(prompts)}
            iterator = as_completed(futures)
            if use_tqdm:
                iterator = tqdm(iterator, total=len(prompts), desc=progress_label or "API Inference", unit="req")
            for future in iterator:
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                if progress_log_path and (completed == 1 or completed % 25 == 0 or completed == len(prompts)):
                    _write_progress()
        if progress_log_path:
            _write_progress()
        return results


def _score_outputs(records: list[dict[str, Any]], outputs: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    stats = {
        "overall": {
            "correct": 0,
            "total": 0,
            "answered": 0,
            "empty_raw_output": 0,
            "na_predictions": 0,
            "output_chars": 0,
        },
        "bloom": defaultdict(lambda: {"correct": 0, "total": 0}),
        "modality": defaultdict(lambda: {"correct": 0, "total": 0}),
    }

    for i, out in enumerate(outputs):
        record = records[i]
        raw_output = out.outputs[0].text
        # Handle cases where out.outputs[0].text might be None or Error
        if raw_output is None: raw_output = ""
        
        predicted_letter = _extract_answer_letter(raw_output)
        gt_letter = _get_choice_letter(int(record["_gold_index"]))
        is_correct = predicted_letter == gt_letter
        stripped_output = raw_output.strip()

        results.append(
            {
                "qa_id": record.get("qa_id"),
                "bloom": record.get("bloom_level"),
                "is_multimodal": record.get("is_multimodal", False),
                "gt_letter": gt_letter,
                "pred_letter": predicted_letter,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "empty_raw_output": stripped_output == "",
            }
        )

        stats["overall"]["total"] += 1
        stats["overall"]["output_chars"] += len(raw_output)
        if is_correct:
            stats["overall"]["correct"] += 1
        if stripped_output == "":
            stats["overall"]["empty_raw_output"] += 1
        if predicted_letter == "N/A":
            stats["overall"]["na_predictions"] += 1
        else:
            stats["overall"]["answered"] += 1

        bloom = record.get("bloom_level", "Unknown")
        stats["bloom"][bloom]["total"] += 1
        if is_correct:
            stats["bloom"][bloom]["correct"] += 1

        modality = "multimodal" if record.get("is_multimodal") else "text_only"
        stats["modality"][modality]["total"] += 1
        if is_correct:
            stats["modality"][modality]["correct"] += 1

    return results, stats


def _report_stem(model_id: str, run_name: str | None, split: str | None, no_context: bool) -> str:
    if run_name:
        return run_name
    stem = model_id.split("/")[-1]
    if split:
        stem += f"_{split}"
    stem += "_noctx" if no_context else "_ctx"
    return stem


def _write_report(
    output_dir: Path,
    model_id: str,
    raw_records: list[dict[str, Any]],
    records: list[dict[str, Any]],
    skip_reasons: Counter,
    split: str | None,
    no_context: bool,
    run_name: str | None,
    results: list[dict[str, Any]],
    stats: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _report_stem(model_id, run_name, split, no_context)
    report_path = output_dir / f"bench_{stem}.json"

    final_report = {
        "model_id": model_id,
        "n_input_samples": len(raw_records),
        "n_samples": len(records),
        "skipped": dict(skip_reasons),
        "no_context": no_context,
        "split": split,
        "run_name": stem,
        "metrics": {
            "accuracy_overall": stats["overall"]["correct"] / stats["overall"]["total"] if stats["overall"]["total"] > 0 else 0,
            "by_bloom": {k: v["correct"] / v["total"] for k, v in stats["bloom"].items()},
            "by_modality": {k: v["correct"] / v["total"] for k, v in stats["modality"].items()},
        },
        "diagnostics": {
            "answered_count": stats["overall"]["answered"],
            "answered_rate": stats["overall"]["answered"] / stats["overall"]["total"] if stats["overall"]["total"] > 0 else 0,
            "empty_raw_output_count": stats["overall"]["empty_raw_output"],
            "empty_raw_output_rate": stats["overall"]["empty_raw_output"] / stats["overall"]["total"] if stats["overall"]["total"] > 0 else 0,
            "na_prediction_count": stats["overall"]["na_predictions"],
            "na_prediction_rate": stats["overall"]["na_predictions"] / stats["overall"]["total"] if stats["overall"]["total"] > 0 else 0,
            "mean_output_length_chars": stats["overall"]["output_chars"] / stats["overall"]["total"] if stats["overall"]["total"] > 0 else 0,
            "valid_for_interpretation": (
                (stats["overall"]["empty_raw_output"] / stats["overall"]["total"]) < 0.10
                if stats["overall"]["total"] > 0
                else False
            ),
        },
        "details": results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Evaluation complete. Report: {report_path}")
    print(f"--- RESULTS: {model_id} / {'no-context' if no_context else 'with-context'} ---")
    print(f"Overall Accuracy: {final_report['metrics']['accuracy_overall']:.2%}")
    print(f"Answered Rate : {final_report['diagnostics']['answered_rate']:.2%}")
    print(f"Empty Outputs : {final_report['diagnostics']['empty_raw_output_rate']:.2%}")
    for bloom, acc in final_report["metrics"]["by_bloom"].items():
        print(f"  {bloom}: {acc:.2%}")
    for modality, acc in final_report["metrics"]["by_modality"].items():
        print(f"  {modality}: {acc:.2%}")
    if not final_report["diagnostics"]["valid_for_interpretation"]:
        print("[WARN] High empty-output rate detected; do not interpret this run as a benchmark.")
    return report_path


def _run_single_mode(
    llm: Any,
    sampling_params: Any,
    records: list[dict[str, Any]],
    model_id: str,
    output_dir: Path,
    raw_records: list[dict[str, Any]],
    skip_reasons: Counter,
    split: str | None,
    no_context: bool,
    run_name: str | None,
    max_context_chars: int | None,
) -> Path:
    stem = _report_stem(model_id, run_name, split, no_context)
    progress_log_path = output_dir / f"progress_{stem}.json"
    prompts = []
    truncated_count = 0
    for r in records:
        if no_context:
            prompts.append(build_mcq_prompt_no_context(r))
            continue
        original_ctx = " ".join(str(r.get("context_payload", {}).get("text", "")).split())
        truncated_ctx = _truncate_context(original_ctx, max_context_chars)
        if truncated_ctx != original_ctx:
            truncated_count += 1
        prompts.append(build_mcq_prompt(r, max_context_chars=max_context_chars))

    mode_name = "no-context" if no_context else "with-context"

    print(f"[INFO] Running {mode_name} inference for {len(records)} prompts on {model_id}...")
    if not no_context and max_context_chars:
        print(f"[INFO] Context truncation enabled: max_context_chars={max_context_chars} (truncated {truncated_count}/{len(records)} prompts)")
    print(f"[INFO] Progress log: {progress_log_path}")
    start_time = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True
    )
    elapsed = time.perf_counter() - start_time
    print(f"[INFO] {mode_name} complete in {elapsed:.2f}s ({len(records)/elapsed:.2f} samples/s)")

    results, stats = _score_outputs(records, outputs)
    return _write_report(
        output_dir=output_dir,
        model_id=model_id,
        raw_records=raw_records,
        records=records,
        skip_reasons=skip_reasons,
        split=split,
        no_context=no_context,
        run_name=run_name,
        results=results,
        stats=stats,
    )


def run_eval(
    dataset_path: Path,
    model_id: str,
    output_dir: Path,
    limit: int | None = None,
    gpu_util: float = 0.90,
    tensor_parallel: int = 1,
    no_context: bool = False,
    split: str | None = None,
    run_name: str | None = None,
    max_model_len: int = 3072,
    max_num_seqs: int = 32,
    both_modes: bool = False,
    use_api: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    max_context_chars: int | None = None,
) -> None:
    raw_records, records, skip_reasons = _load_evaluable_records(
        dataset_path=dataset_path,
        limit=limit,
        split=split,
    )

    print(f"[INFO] Loaded {len(raw_records)} records from {dataset_path}")
    print(f"[INFO] Evaluable 4-way records: {len(records)}")
    if skip_reasons:
        print(f"[INFO] Skipped records: {dict(skip_reasons)}")
    if not records:
        print("[ERROR] No evaluable 4-way records found after ground-truth resolution.")
        return

    llm = None
    sampling_params = None

    if use_api or "http" in model_id:
        print(f"[INFO] Using API mode for {model_id}")
        _api_base = api_base or "http://10.9.5.18:1234/v1"
        _api_key = api_key or "sk-tqa-benchmark-2026-v1"
        client = _init_api_client(_api_base, _api_key)
        if client:
            llm = APIModelWrapper(client, model_id)
            # sampling_params not strictly needed for APIModelWrapper but kept for signature
            sampling_params = type('obj', (object,), {'max_tokens': 4, 'temperature': 0.0}) 
        else:
            return
    else:
        llm, sampling_params = _init_llm(
            model_id=model_id,
            gpu_util=gpu_util,
            tensor_parallel=tensor_parallel,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
        )

    if llm is None:
        print("[ERROR] Failed to initialize LLM (local or API)")
        return

    if both_modes:
        ctx_stem = _report_stem(model_id, run_name, split, False)
        noctx_stem = _report_stem(model_id, run_name, split, True)
        if run_name:
            ctx_stem = f"{run_name}_ctx"
            noctx_stem = f"{run_name}_noctx"
        _run_single_mode(
            llm=llm,
            sampling_params=sampling_params,
            records=records,
            model_id=model_id,
            output_dir=output_dir,
            raw_records=raw_records,
            skip_reasons=skip_reasons,
            split=split,
            no_context=False,
            run_name=ctx_stem,
            max_context_chars=max_context_chars,
        )
        _run_single_mode(
            llm=llm,
            sampling_params=sampling_params,
            records=records,
            model_id=model_id,
            output_dir=output_dir,
            raw_records=raw_records,
            skip_reasons=skip_reasons,
            split=split,
            no_context=True,
            run_name=noctx_stem,
            max_context_chars=max_context_chars,
        )
        return

    _run_single_mode(
        llm=llm,
        sampling_params=sampling_params,
        records=records,
        model_id=model_id,
        output_dir=output_dir,
        raw_records=raw_records,
        skip_reasons=skip_reasons,
        split=split,
        no_context=no_context,
        run_name=run_name,
        max_context_chars=max_context_chars,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot MCQ Benchmark for DHH-LegalQA.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="vLLM model ID")
    parser.add_argument("--out-dir", type=Path, default=Path("research/results/benchmarks"), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--gpu-util", type=float, default=0.90, help="VRAM reservation")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--max-model-len", type=int, default=3072, help="vLLM max model length")
    parser.add_argument("--max-num-seqs", type=int, default=32, help="vLLM max concurrent sequences")
    parser.add_argument("--no-context", action="store_true", help="Evaluate without context text")
    parser.add_argument("--both-modes", action="store_true", help="Run both with-context and no-context in one model load")
    parser.add_argument("--split", type=str, default=None, help="Optional split filter, e.g. test")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output stem override")
    parser.add_argument("--use-api", action="store_true", help="Use OpenAI-compatible API instead of local vLLM")
    parser.add_argument("--api-base", type=str, default=os.getenv("BENCHMARK_API_BASE", "http://localhost:1234/v1"), help="API base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("BENCHMARK_API_KEY", "sk-tqa-benchmark-2026-v1"), help="API key")
    parser.add_argument("--max-context-chars", type=int, default=2000, help="Maximum characters from context to include in with-context prompts")
    
    args = parser.parse_args()
    use_api = args.use_api
    # Force GPU only for 'train' split as per project policy
    if args.split == "train" and use_api:
        print("[WARN] Forcing GPU-only (local vLLM) for 'train' split. Disabling --use-api.")
        use_api = False

    run_eval(
        args.dataset,
        args.model,
        args.out_dir,
        args.limit,
        args.gpu_util,
        tensor_parallel=args.tensor_parallel,
        no_context=args.no_context,
        split=args.split,
        run_name=args.run_name,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        both_modes=args.both_modes,
        use_api=use_api,
        api_base=args.api_base,
        api_key=args.api_key,
        max_context_chars=args.max_context_chars,
    )
