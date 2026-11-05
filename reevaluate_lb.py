"""
Force re-evaluate all models that already have results on disk.

For each existing results_*.jsonl file, re-runs the full benchmark with
force_rerun=True.  If the new evaluation succeeds and produces a non-empty
DataFrame the existing file is replaced; if it fails the old file is kept
untouched.

Run BEFORE upload_repacked_results.py to refresh all metrics on disk.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

from tqdm import tqdm

from env import (
    DATA_DIR,
    DEFAULT_MODEL_EVALUATIONS_PATH,
    DEFAULT_RESULTS_PATH,
)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _read_first_line_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        line = fh.readline().strip()
    if not line:
        return None
    return json.loads(line)


def discover_evaluated_models(evaluations_dir: Path, verbose: bool = False):
    """
    Scan all results_*.jsonl files (loose) and return a list of task dicts:
        {model, revision, subfolder, _target_mode, results_path}

    Models that appear only inside batch_*.zip archives but not as loose files
    are ignored (upload_repacked_results.py handles zips; loose files are the
    canonical source of truth after fix_ebpb / reevaluation passes).
    """
    tasks = []
    seen_keys = set()

    for result_file in sorted(evaluations_dir.glob("results_*.jsonl")):
        try:
            data = _read_first_line_json(result_file)
            if data is None:
                if verbose:
                    print(f"Warning: empty file {result_file.name}, skipping")
                continue

            model = data.get("model")
            if not model:
                if verbose:
                    print(f"Warning: no 'model' key in {result_file.name}, skipping")
                continue

            revision = data.get("revision") or "main"
            subfolder = data.get("subfolder") or None
            _target_mode = bool(data.get("_target_mode", False))

            # Deduplicate by (model, revision, subfolder, _target_mode)
            key = (model, revision, subfolder, _target_mode)
            if key in seen_keys:
                if verbose:
                    print(f"Warning: duplicate key {key} from {result_file.name}, skipping")
                continue
            seen_keys.add(key)

            tasks.append(
                {
                    "model": model,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_target_mode": _target_mode,
                    "results_path": result_file,
                }
            )
        except Exception as exc:
            if verbose:
                print(f"Warning: could not parse {result_file.name}: {exc}")
            continue

    return tasks


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _cleanup_model_cache(model_name: str) -> str:
    """Delete all cached revisions for model_name from the HF hub cache."""
    try:
        from huggingface_hub import scan_cache_dir

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
        return f"Cache cleared for {model_name} (freed {delete_strategy.expected_freed_size_str})"
    except Exception as exc:
        return f"Cache cleanup failed for {model_name}: {exc}"


def _worker(task_with_ctx):
    """
    Runs in a subprocess (mp.Pool worker).

    task_with_ctx is a tuple:
        (task_dict, benchmark_dir, benchmark_file_hashes, tokenizer_hash_cache,
         batch_size, verbose, progress_info)
    """
    (
        task,
        benchmark_dir,
        benchmark_file_hashes,
        tokenizer_hash_cache,
        batch_size,
        verbose,
        progress_info,
    ) = task_with_ctx

    current, total = progress_info
    model = task["model"]
    revision = task["revision"]
    subfolder = task["subfolder"]
    _target_mode = task["_target_mode"]
    results_path: Path = task["results_path"]

    pid = os.getpid()
    print(
        f"[{pid}] [{current}/{total}] Starting {model} "
        f"(rev={revision}, sub={subfolder}, target={_target_mode})"
    )

    try:
        from tokenizer_evaluate import run_benchmark

        df = run_benchmark(
            model_name=model,
            revision=revision,
            subfolder=subfolder,
            _target_mode=_target_mode,
            benchmark_dir=benchmark_dir,
            langs=None,
            verbose=verbose,
            save_results=False,      # We handle file replacement ourselves
            force_rerun=True,
            benchmark_file_hashes=benchmark_file_hashes,
            tokenizer_hash_cache=tokenizer_hash_cache,
            upload_results=False,
            trust_remote_code=True,
            use_batched_evaluation=True,
            batch_size=batch_size,
        )

        if df is None or df.empty:
            raise ValueError("run_benchmark returned an empty DataFrame")

        # Atomic-ish replacement: write to .tmp then rename
        tmp_path = results_path.with_suffix(".jsonl.tmp")
        df.to_json(tmp_path, lines=True, orient="records", force_ascii=False)
        tmp_path.rename(results_path)

        cleanup_msg = _cleanup_model_cache(model)
        print(f"[{pid}] [{current}/{total}] Completed {model}" + (f" | {cleanup_msg}" if cleanup_msg else ""))
        return {"model": model, "success": True, "error": None}

    except Exception as exc:
        error_msg = str(exc)
        print(f"[{pid}] [{current}/{total}] FAILED {model}: {error_msg}")
        if verbose:
            traceback.print_exc()
        # tmp file may exist if crash happened after write but before rename
        tmp_path = results_path.with_suffix(".jsonl.tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        cleanup_msg = _cleanup_model_cache(model)
        if cleanup_msg:
            print(f"[{pid}] [{current}/{total}] {cleanup_msg}")
        return {"model": model, "success": False, "error": error_msg}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Force re-evaluate all models that already have results on disk. "
            "Replaces each results file only when evaluation succeeds."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run: list all models that would be re-evaluated
  uv run reevaluate_lb.py --dry-run

  # Single process (safe default)
  uv run reevaluate_lb.py

  # Parallel with 4 workers
  uv run reevaluate_lb.py --num-processes 4

  # Parallel, verbose
  uv run reevaluate_lb.py --num-processes 4 --verbose
""",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for batched evaluation (default: 1000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output per model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List all models that would be re-evaluated without running anything",
    )
    args = parser.parse_args()

    # Disable HF tokenizer parallelism when using multiple processes to avoid
    # fork-safety issues (same pattern as init_lb.py)
    if args.num_processes > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    evaluations_dir = Path(DATA_DIR) / DEFAULT_RESULTS_PATH / DEFAULT_MODEL_EVALUATIONS_PATH

    if not evaluations_dir.exists():
        print(f"Evaluations directory not found: {evaluations_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Discover models
    # ------------------------------------------------------------------
    print(f"Scanning {evaluations_dir} for evaluated models...")
    tasks = discover_evaluated_models(evaluations_dir, verbose=args.verbose)
    print(f"Found {len(tasks)} models to re-evaluate.")

    if not tasks:
        print("Nothing to do.")
        sys.exit(0)

    if args.dry_run:
        print("\n[dry-run] Would re-evaluate the following models:")
        for t in tasks:
            print(
                f"  {t['model']}  rev={t['revision']}  "
                f"sub={t['subfolder']}  target={t['_target_mode']}  "
                f"-> {t['results_path'].name}"
            )
        sys.exit(0)

    # ------------------------------------------------------------------
    # 2. Download benchmark data once (in main process)
    # ------------------------------------------------------------------
    print("Downloading benchmark and results repos...")
    from tokenizer_evaluate import download_repos_once

    benchmark_dir, results_dir, benchmark_file_hashes, tokenizer_hash_cache = (
        download_repos_once()
    )
    print("Repos ready.")

    # ------------------------------------------------------------------
    # 3. Build worker args
    # ------------------------------------------------------------------
    tasks_with_ctx = [
        (
            task,
            benchmark_dir,
            benchmark_file_hashes,
            tokenizer_hash_cache,
            args.batch_size,
            args.verbose,
            (i + 1, len(tasks)),
        )
        for i, task in enumerate(tasks)
    ]

    # ------------------------------------------------------------------
    # 4. Run evaluations
    # ------------------------------------------------------------------
    start_time = time.time()

    if args.num_processes <= 1:
        results = []
        for i, task_ctx in enumerate(
            tqdm(tasks_with_ctx, desc="Re-evaluating tokenizers", unit="model"), 1
        ):
            result = _worker(task_ctx)
            results.append(result)
    else:
        with mp.Pool(processes=args.num_processes) as pool:
            results = []
            with tqdm(
                total=len(tasks_with_ctx),
                desc="Re-evaluating tokenizers",
                unit="model",
            ) as pbar:
                for result in pool.imap(_worker, tasks_with_ctx):
                    results.append(result)
                    pbar.update(1)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print(f"COMPLETED: {successful}/{len(results)} successful, {failed} failed")
    print(f"Total time: {total_time:.1f}s  ({total_time / max(len(results), 1):.1f}s avg)")
    print(f"{'='*60}")

    if failed > 0:
        print("\nFailed models:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['model']}: {r['error']}")


if __name__ == "__main__":
    main()
