"""
Compact loose JSONL evaluation results into 500-file zip batches.

Run manually when the number of loose files grows large:
    uv run python compact_results.py
    uv run python compact_results.py --not_upload   # local only
    uv run python compact_results.py --batch_size 1000
"""
import argparse
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, CommitOperationAdd, CommitOperationDelete

from env import HF_REPO_RESULTS, DATA_DIR

DEFAULT_RESULTS_PATH = "results"
DEFAULT_MODEL_EVALUATIONS_PATH = "evaluations"
BATCH_SIZE = 500


def _files_in_existing_zips(evaluations_dir: Path) -> set[str]:
    """Return the set of JSONL filenames already packed inside batch zips."""
    packed = set()
    for zf_path in sorted(evaluations_dir.glob("batch_*.zip")):
        with zipfile.ZipFile(zf_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".jsonl"):
                    packed.add(name)
    return packed


def _next_batch_number(evaluations_dir: Path) -> int:
    """Determine the next sequential batch number."""
    existing = sorted(evaluations_dir.glob("batch_*.zip"))
    if not existing:
        return 1
    last = existing[-1].stem  # e.g. "batch_017"
    return int(last.split("_")[1]) + 1


def compact_results(upload_to_hub: bool = True, batch_size: int = BATCH_SIZE):
    results_dir = Path(DATA_DIR) / DEFAULT_RESULTS_PATH

    print("Downloading results repository...")
    snapshot_download(
        repo_id=HF_REPO_RESULTS,
        local_dir=results_dir,
        repo_type="dataset",
        allow_patterns=[
            f"{DEFAULT_MODEL_EVALUATIONS_PATH}/batch_*.zip",
            f"{DEFAULT_MODEL_EVALUATIONS_PATH}/*.jsonl",
        ],
        max_workers=4,
    )

    evaluations_dir = results_dir / DEFAULT_MODEL_EVALUATIONS_PATH
    if not evaluations_dir.exists():
        print(f"Evaluations directory not found: {evaluations_dir}")
        return

    packed = _files_in_existing_zips(evaluations_dir)
    print(f"Already packed in existing zips: {len(packed)} files")

    loose_files = sorted(
        f for f in evaluations_dir.glob("results_*.jsonl")
        if f.name not in packed
    )
    print(f"Loose JSONL files to consider: {len(loose_files)}")

    if len(loose_files) < batch_size:
        print(f"Fewer than {batch_size} loose files — nothing to compact.")
        return

    full_batches = len(loose_files) // batch_size
    next_num = _next_batch_number(evaluations_dir)

    api = HfApi()
    add_ops: list[CommitOperationAdd] = []
    delete_ops: list[CommitOperationDelete] = []

    for i in range(full_batches):
        batch_files = loose_files[i * batch_size : (i + 1) * batch_size]
        batch_name = f"batch_{next_num + i:03d}.zip"
        zip_path = evaluations_dir / batch_name

        print(f"Creating {batch_name} with {len(batch_files)} files...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in batch_files:
                zf.write(fp, fp.name)

        if upload_to_hub:
            add_ops.append(CommitOperationAdd(
                path_in_repo=f"{DEFAULT_MODEL_EVALUATIONS_PATH}/{batch_name}",
                path_or_fileobj=str(zip_path),
            ))
            for fp in batch_files:
                delete_ops.append(CommitOperationDelete(
                    path_in_repo=f"{DEFAULT_MODEL_EVALUATIONS_PATH}/{fp.name}",
                ))

    remaining = len(loose_files) - full_batches * batch_size
    print(f"Created {full_batches} zip(s). {remaining} loose file(s) left for next run.")

    if upload_to_hub and (add_ops or delete_ops):
        print(f"Uploading {len(add_ops)} zip(s) and deleting {len(delete_ops)} loose file(s) on Hub...")
        try:
            api.create_commit(
                repo_id=HF_REPO_RESULTS,
                operations=add_ops + delete_ops,
                commit_message=f"chore: compact {full_batches * batch_size} JSONL files into {full_batches} batch zip(s)",
                repo_type="dataset",
            )
            print("Upload complete.")
        except Exception as e:
            print(f"Upload failed: {e}")
    elif upload_to_hub:
        print("No changes to upload.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compact loose JSONL results into zip batches.")
    parser.add_argument("--not_upload", action="store_true", help="Skip uploading to Hub.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Files per zip (default {BATCH_SIZE}).")
    args = parser.parse_args()

    compact_results(upload_to_hub=not args.not_upload, batch_size=args.batch_size)
