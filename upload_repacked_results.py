"""
Repack all batch zip archives with updated JSONL files already on disk,
then commit both the repacked zips and any loose JSONL files to the
Hugging Face dataset repository.

Run AFTER fix_ebpb_metrics.py has updated the loose JSONL files but the
upload step failed, so we don't need to redo any metric computation.
"""

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from tqdm import tqdm

from env import (
    DATA_DIR,
    DEFAULT_MODEL_EVALUATIONS_PATH,
    DEFAULT_RESULTS_PATH,
    HF_REPO_RESULTS,
)
from tokenizer_evaluate import update_files_in_zip


def build_file_to_zip(evaluations_dir: Path) -> dict[str, Path]:
    """Map each JSONL member name -> its containing zip path."""
    file_to_zip: dict[str, Path] = {}
    for zip_path in sorted(evaluations_dir.glob("batch_*.zip")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".jsonl"):
                    file_to_zip[member] = zip_path
    return file_to_zip


def repack_zips(evaluations_dir: Path, file_to_zip: dict[str, Path], dry_run: bool) -> list[CommitOperationAdd]:
    """Re-pack every affected zip with the updated on-disk JSONL files."""
    zip_to_members: dict[Path, dict[str, Path]] = {}
    for member_name, zip_path in file_to_zip.items():
        local_file = evaluations_dir / member_name
        if not local_file.exists():
            continue
        zip_to_members.setdefault(zip_path, {})[member_name] = local_file

    operations: list[CommitOperationAdd] = []
    for zip_path, updated_members in tqdm(zip_to_members.items(), desc="Re-packing zips"):
        print(f"  {zip_path.name}: {len(updated_members)} file(s)")
        if not dry_run:
            update_files_in_zip(zip_path, updated_members)
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{DEFAULT_MODEL_EVALUATIONS_PATH}/{zip_path.name}",
                path_or_fileobj=str(zip_path),
            )
        )
    return operations


def collect_loose_operations(evaluations_dir: Path, file_to_zip: dict[str, Path]) -> list[CommitOperationAdd]:
    """Return CommitOperationAdd for every JSONL not contained in any zip."""
    zipped_names = set(file_to_zip.keys())
    loose_files = sorted(
        f for f in evaluations_dir.glob("results_*.jsonl") if f.name not in zipped_names
    )
    print(f"Found {len(loose_files)} loose JSONL file(s) not in any zip.")
    return [
        CommitOperationAdd(
            path_in_repo=f"{DEFAULT_MODEL_EVALUATIONS_PATH}/{f.name}",
            path_or_fileobj=str(f),
        )
        for f in loose_files
    ]


def upload(
    dry_run: bool = False,
    commit_message: str = "chore: Recompute EBPB metric (rate-distortion form)",
):
    evaluations_dir = Path(DATA_DIR) / DEFAULT_RESULTS_PATH / DEFAULT_MODEL_EVALUATIONS_PATH

    if not evaluations_dir.exists():
        print(f"Evaluations directory not found: {evaluations_dir}")
        return

    print(f"Scanning zip archives in {evaluations_dir} ...")
    file_to_zip = build_file_to_zip(evaluations_dir)
    print(f"  {len(set(file_to_zip.values()))} zip(s) covering {len(file_to_zip)} JSONL member(s).")

    zip_operations = repack_zips(evaluations_dir, file_to_zip, dry_run=dry_run)
    loose_operations = collect_loose_operations(evaluations_dir, file_to_zip)

    all_operations = zip_operations + loose_operations
    print(
        f"\nTotal operations: {len(zip_operations)} zip(s) + {len(loose_operations)} loose file(s)"
        f" = {len(all_operations)} commit operation(s)."
    )

    if not all_operations:
        print("Nothing to upload.")
        return

    if dry_run:
        print("\n[dry-run] Would upload the following paths:")
        for op in all_operations:
            print(f"  {op.path_in_repo}")
        return

    api = HfApi()
    print(f"\nUploading to {HF_REPO_RESULTS} ...")
    try:
        api.create_commit(
            repo_id=HF_REPO_RESULTS,
            operations=all_operations,
            commit_message=commit_message,
            repo_type="dataset",
        )
        print("Upload complete.")
    except Exception as e:
        print(f"Upload error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repack batch zips and upload updated results to Hugging Face Hub."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without repacking or uploading.",
    )
    parser.add_argument(
        "--commit-message",
        default="chore: Recompute EBPB metric (rate-distortion form)",
        help="Git commit message for the Hub commit.",
    )
    args = parser.parse_args()

    upload(dry_run=args.dry_run, commit_message=args.commit_message)
