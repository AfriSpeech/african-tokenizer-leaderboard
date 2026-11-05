"""
Recompute EBPB and byte-fidelity metrics for all stored evaluation JSONL files.

For each existing result JSONL it:
  1. Loads the tokenizer once.
  2. For every standard/parity row, loads the matching benchmark .jsonl.gz,
     tokenizes, decodes, and computes byte-edit stats.
  3. Patches only the 6 columns into the row dict without touching anything
     else.

No HF API calls, no vocab hashing, no near-duplicate detection.
"""

import argparse
import importlib
import json
import math
import multiprocessing
import os
import traceback
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, scan_cache_dir
from tqdm import tqdm
from transformers import AutoTokenizer

from env import DEFAULT_MODEL_EVALUATIONS_PATH, HF_REPO_RESULTS
from tokenizer_evaluate import (
    download_repos,
    extract_result_zips,
    load_dataset_meta,
    update_files_in_zip,
    yield_jsonl_gz as _yield_jsonl_gz,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

METRIC_COLUMNS = [
    "byte_edit_distance_total",
    "byte_edit_denominator_total",
    "decoded_bytes_total",
    "byte_fidelity",
    "effective_bytes_per_token",
    "ebpb",
]

DEFAULT_BATCH_SIZE = 10_000

# ---------------------------------------------------------------------------
# Byte-fidelity metric helpers (mirrors tokenizer_evaluate.py, no imports)
# ---------------------------------------------------------------------------

_LEVENSHTEIN_DISTANCE = None


def _get_levenshtein_distance():
    global _LEVENSHTEIN_DISTANCE
    if _LEVENSHTEIN_DISTANCE is None:
        mod = importlib.import_module("rapidfuzz.distance.Levenshtein")
        _LEVENSHTEIN_DISTANCE = mod.distance
    return _LEVENSHTEIN_DISTANCE


def _byte_edit_stats(original_text: str, decoded_text: str):
    ob = original_text.encode("utf-8")
    db = decoded_text.encode("utf-8")
    distance = _get_levenshtein_distance()(ob, db)
    denominator = max(len(ob), len(db), 1)
    return distance, denominator, len(db)


def _calculate_byte_fidelity(distance_total: int, denominator_total: int) -> float:
    if denominator_total <= 0:
        return 0.0
    return round(max(1.0 - distance_total / denominator_total, 0.0), 5)


def _default_format(value: float, precision: int = 3) -> float:
    return round(value, precision) if value > 0 else 0.0


def _calculate_effective_bytes_per_token(compression_bytes: float, byte_fidelity: float) -> float:
    return _default_format(compression_bytes * byte_fidelity)


def _calculate_ebpb(total_tokens, total_bytes, byte_edit_distance_total, freq_cardinality) -> float:
    """Calculate EBPB using the rate-distortion formula. Lower is better.

    EBPB = (T * log2(V_obs) + 8 * D_byte) / B
    """
    if not total_tokens or not total_bytes or not freq_cardinality or freq_cardinality < 2:
        return None
    obs = max(int(freq_cardinality), 2)
    value = (total_tokens * math.log2(obs) + 8 * byte_edit_distance_total) / total_bytes
    return _default_format(max(value, 0))


# ---------------------------------------------------------------------------
# Benchmark-file helpers
# ---------------------------------------------------------------------------
def _build_benchmark_lookup(benchmark_dir: Path) -> dict:
    """Build a dict mapping (lang, domain_name) -> Path for all benchmark files."""
    meta_path = benchmark_dir / "dataset_meta.yaml"
    if not meta_path.exists():
        return {}
    meta = load_dataset_meta(meta_path)
    lookup = {}
    for lang, lang_info in meta.items():
        if not isinstance(lang_info, dict):
            continue
        for domain in lang_info.get("domains") or []:
            lookup[(lang, domain["name"])] = benchmark_dir / domain["filepath"]
        for parity in lang_info.get("parities") or []:
            lookup[(lang, parity["name"])] = benchmark_dir / parity["filepath"]
            # Reference lang is the first path component of reference_filepath (e.g. "en/parity.jsonl.gz" -> "en")
            ref_lang = Path(parity["reference_filepath"]).parts[0]
            lookup[(ref_lang, parity["name"])] = benchmark_dir / parity["reference_filepath"]
    return lookup


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

_BENCHMARK_DIR: Path = None
_BENCHMARK_LOOKUP: dict = None
_BATCH_SIZE: int = DEFAULT_BATCH_SIZE
_TRUST_REMOTE_CODE: bool = True


def _init_worker(benchmark_dir_str, batch_size, trust_remote_code):
    global _BENCHMARK_DIR, _BENCHMARK_LOOKUP, _BATCH_SIZE, _TRUST_REMOTE_CODE
    _BENCHMARK_DIR = Path(benchmark_dir_str)
    _BENCHMARK_LOOKUP = _build_benchmark_lookup(_BENCHMARK_DIR)
    _BATCH_SIZE = batch_size
    _TRUST_REMOTE_CODE = trust_remote_code


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _load_tokenizer(model_name: str, revision: str, subfolder, _target_mode: bool):
    kwargs = {"trust_remote_code": _TRUST_REMOTE_CODE, "revision": revision}
    if subfolder:
        kwargs["subfolder"] = subfolder
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, **kwargs)

    if _target_mode:
        if hasattr(tok, "_switch_to_target_mode"):
            tok._switch_to_target_mode()
        elif hasattr(tok, "as_target_tokenizer"):
            tok.as_target_tokenizer()

    tok.model_max_length = int(1e9)
    return tok


def _cleanup_model_cache(model_name: str) -> str:
    """Delete all cached revisions for model_name from the HF hub cache."""
    try:
        cache_info = scan_cache_dir()
        to_delete = [
            revision.commit_hash
            for repo in cache_info.repos
            if repo.repo_id == model_name
            for revision in repo.revisions
        ]
        if not to_delete:
            return ""
        delete_strategy = cache_info.delete_revisions(*to_delete)
        delete_strategy.execute()
        return f"  - Cache cleared for {model_name} (freed {delete_strategy.expected_freed_size_str})"
    except Exception as e:
        return f"  - Cache cleanup failed for {model_name}: {e}"


# ---------------------------------------------------------------------------
# Per-row metric computation
# ---------------------------------------------------------------------------

def _compute_row_metrics(tokenizer, texts: list[str], compression_bytes: float, vocab_size, _target_mode: bool) -> dict:
    """Tokenize + decode texts and return byte-fidelity metric values plus raw token counts."""
    from collections import Counter as _Counter
    if not texts:
        return {col: 0 for col in METRIC_COLUMNS}

    # Tokenize
    if not _target_mode:
        encoding = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="np",
        )
    else:
        encoding = tokenizer(
            text_target=texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="np",
        )

    decoded_texts = tokenizer.batch_decode(
        [ids.tolist() for ids in encoding.input_ids]
    )

    byte_edit_distance_total = 0
    byte_edit_denominator_total = 0
    decoded_bytes_total = 0
    total_tokens = 0
    token_counter: _Counter = _Counter()

    for ids, orig, decoded in zip(encoding.input_ids, texts, decoded_texts):
        ids_list = ids.tolist()
        token_counter.update(ids_list)
        total_tokens += len(ids_list)
        dist, denom, dec_len = _byte_edit_stats(orig, decoded)
        byte_edit_distance_total += dist
        byte_edit_denominator_total += denom
        decoded_bytes_total += dec_len

    byte_fidelity = _calculate_byte_fidelity(byte_edit_distance_total, byte_edit_denominator_total)
    effective_bytes_per_token = _calculate_effective_bytes_per_token(compression_bytes, byte_fidelity)

    return {
        "byte_edit_distance_total": int(byte_edit_distance_total),
        "byte_edit_denominator_total": int(byte_edit_denominator_total),
        "decoded_bytes_total": int(decoded_bytes_total),
        "byte_fidelity": byte_fidelity,
        "effective_bytes_per_token": effective_bytes_per_token,
        # Raw values needed to compute EBPB after aggregation
        "_total_tokens": total_tokens,
        "_token_counter": token_counter,
    }


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def _all_metrics_present(rows: list[dict]) -> bool:
    """Return True if every row that should have metrics already has all 6 columns set."""
    checkable = [r for r in rows if not r.get("reference", False)]
    if not checkable:
        return False
    for row in checkable:
        for col in METRIC_COLUMNS:
            v = row.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return False
    return True


def process_result_file(args):
    file_path_str, upload_to_hub, evaluations_path, force = args
    file_path = Path(file_path_str)
    logs = []
    local_operations = []
    rows = []
    model_name = None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if not rows:
            logs.append(f"Skipping {file_path.name}: empty file.")
            return local_operations, logs

        if not force and _all_metrics_present(rows):
            logs.append(f"  - {file_path.name}: already has metrics, skipping (use --force to recompute).")
            return local_operations, logs

        first = rows[0]
        model_name = first.get("model")
        revision = first.get("revision") or "main"
        subfolder_raw = first.get("subfolder")
        subfolder = subfolder_raw if subfolder_raw and str(subfolder_raw) not in ("", "null", "None") else None
        _target_mode = bool(first.get("_target_mode", False))

        if not model_name:
            logs.append(f"Skipping {file_path.name}: missing model name.")
            return local_operations, logs

        logs.append(f"Processing {model_name} ({file_path.name})")

        try:
            tokenizer = _load_tokenizer(model_name, revision, subfolder, _target_mode)
        except Exception as e:
            logs.append(f"  - Could not load tokenizer for {model_name}: {e}")
            return local_operations, logs

        # Build a lookup: (lang, domain) -> benchmark texts
        # Load each unique benchmark file at most once per worker call
        texts_cache: dict[tuple, list[str]] = {}

        def _get_texts(lang: str, domain: str) -> list[str]:
            key = (lang, domain)
            if key in texts_cache:
                return texts_cache[key]
            bpath = _BENCHMARK_LOOKUP.get(key)
            if bpath is None or not bpath.exists():
                texts_cache[key] = []
                return []
            texts = []
            for entry in _yield_jsonl_gz(bpath):
                text = entry["text"].strip()
                if text:
                    texts.append(text)
            texts_cache[key] = texts
            return texts

        # Group rows by (lang, domain) to avoid re-encoding the same file
        from collections import defaultdict
        key_to_row_indices: dict[tuple, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            key_to_row_indices[(row.get("lang"), row.get("domain"))].append(i)

        from collections import Counter as _Counter
        updated = False
        for (lang, domain), indices in key_to_row_indices.items():
            representative = rows[indices[0]]
            compression_bytes = float(representative.get("compression_bytes") or 0)
            total_bytes = float(representative.get("total_bytes") or 0)

            texts = _get_texts(lang, domain)
            if not texts:
                logs.append(f"  - No texts found for ({lang}, {domain}), skipping rows.")
                continue

            # Process in batches to respect memory
            all_texts_batches = [texts[i:i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]

            agg = {
                "byte_edit_distance_total": 0,
                "byte_edit_denominator_total": 0,
                "decoded_bytes_total": 0,
                "_total_tokens": 0,
                "_token_counter": _Counter(),
            }
            for batch in all_texts_batches:
                batch_metrics = _compute_row_metrics(tokenizer, batch, compression_bytes, None, _target_mode)
                agg["byte_edit_distance_total"] += batch_metrics["byte_edit_distance_total"]
                agg["byte_edit_denominator_total"] += batch_metrics["byte_edit_denominator_total"]
                agg["decoded_bytes_total"] += batch_metrics["decoded_bytes_total"]
                agg["_total_tokens"] += batch_metrics["_total_tokens"]
                agg["_token_counter"] += batch_metrics["_token_counter"]

            byte_fidelity = _calculate_byte_fidelity(agg["byte_edit_distance_total"], agg["byte_edit_denominator_total"])
            effective_bytes_per_token = _calculate_effective_bytes_per_token(compression_bytes, byte_fidelity)
            freq_cardinality = len(agg["_token_counter"])
            ebpb = _calculate_ebpb(agg["_total_tokens"], total_bytes, agg["byte_edit_distance_total"], freq_cardinality)

            final_metrics = {
                "byte_edit_distance_total": agg["byte_edit_distance_total"],
                "byte_edit_denominator_total": agg["byte_edit_denominator_total"],
                "decoded_bytes_total": agg["decoded_bytes_total"],
                "byte_fidelity": byte_fidelity,
                "effective_bytes_per_token": effective_bytes_per_token,
                "ebpb": ebpb,
            }

            for i in indices:
                rows[i].update(final_metrics)
                updated = True

        if not updated:
            logs.append(f"  - {file_path.name}: nothing to update.")
            cleanup_msg = _cleanup_model_cache(model_name)
            if cleanup_msg:
                logs.append(cleanup_msg)
            return local_operations, logs

        with open(file_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        cleanup_msg = _cleanup_model_cache(model_name)
        if cleanup_msg:
            logs.append(cleanup_msg)

        if upload_to_hub:
            path_in_repo = f"{evaluations_path}/{file_path.name}"
            local_operations.append(
                CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(file_path))
            )

        logs.append(f"  - Updated {file_path.name} ({len(rows)} rows)")

    except Exception:
        logs.append(f"Failed to update {file_path.name}:\n{traceback.format_exc()}")
        if model_name:
            cleanup_msg = _cleanup_model_cache(model_name)
            if cleanup_msg:
                logs.append(cleanup_msg)

    return local_operations, logs


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def backfill_ebpb_metrics(
    upload_to_hub: bool = True,
    num_processes: int = 60,
    batch_size: int = DEFAULT_BATCH_SIZE,
    trust_remote_code: bool = True,
    force: bool = False,
):
    print("Downloading benchmark and results repos...")
    benchmark_dir, results_dir = download_repos()
    evaluations_dir = results_dir / DEFAULT_MODEL_EVALUATIONS_PATH
    file_to_zip = extract_result_zips(evaluations_dir)

    if not evaluations_dir.exists():
        print(f"Error: evaluations directory not found at {evaluations_dir}")
        return

    result_files = list(evaluations_dir.glob("results_*.jsonl"))
    if not result_files:
        print(f"No result files found in {evaluations_dir}")
        return

    print(f"Processing {len(result_files)} files with {num_processes} workers...")

    worker_args = [
        (str(fp), upload_to_hub, DEFAULT_MODEL_EVALUATIONS_PATH, force)
        for fp in result_files
    ]

    per_file_operations = []
    with multiprocessing.Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(str(benchmark_dir), batch_size, trust_remote_code),
    ) as pool:
        with tqdm(total=len(worker_args), desc="Recomputing EBPB metrics") as pbar:
            for file_ops, file_logs in pool.imap_unordered(process_result_file, worker_args):
                for log in file_logs:
                    tqdm.write(log)
                per_file_operations.extend(file_ops)
                pbar.update()

    # Repack affected zips
    zip_updates: dict = {}
    loose_operations = []
    for op in per_file_operations:
        fname = Path(op.path_in_repo).name
        if fname in file_to_zip:
            zip_path = file_to_zip[fname]
            zip_updates.setdefault(zip_path, {})[fname] = Path(op.path_or_fileobj)
        else:
            loose_operations.append(op)

    zip_operations = []
    for zip_path, updated_members in zip_updates.items():
        print(f"Re-packing {zip_path.name} ({len(updated_members)} file(s))...")
        update_files_in_zip(zip_path, updated_members)
        zip_operations.append(
            CommitOperationAdd(
                path_in_repo=f"{DEFAULT_MODEL_EVALUATIONS_PATH}/{zip_path.name}",
                path_or_fileobj=str(zip_path),
            )
        )

    all_operations = loose_operations + zip_operations
    if upload_to_hub and all_operations:
        api = HfApi()
        print(f"\nUploading {len(zip_operations)} zip(s) and {len(loose_operations)} loose file(s) to Hub...")
        try:
            api.create_commit(
                repo_id=HF_REPO_RESULTS,
                operations=all_operations,
                commit_message="chore: Recompute EBPB metric (rate-distortion form)",
                repo_type="dataset",
            )
            print("Upload complete.")
        except Exception as e:
            print(f"Upload error: {e}")
    elif upload_to_hub:
        print("\nNo files changed, nothing to upload.")

    print("\nBackfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute byte-fidelity and EBPB metrics for all evaluated models.")
    parser.add_argument("--not_upload", action="store_true", help="Skip uploading to Hugging Face Hub.")
    parser.add_argument("--num_processes", type=int, default=60, help="Number of parallel worker processes.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Tokenizer batch size per domain.")
    parser.add_argument("--no_trust_remote_code", action="store_true", help="Do not trust remote code when loading tokenizers.")
    parser.add_argument("--force", action="store_true", help="Re-process files even if EBPB metrics are already present (recommended to recompute with new formula).")
    args = parser.parse_args()

    backfill_ebpb_metrics(
        upload_to_hub=not args.not_upload,
        num_processes=args.num_processes,
        batch_size=args.batch_size,
        trust_remote_code=not args.no_trust_remote_code,
        force=args.force,
    )
