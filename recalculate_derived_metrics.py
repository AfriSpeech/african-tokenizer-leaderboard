"""
Recalculate all metrics that can be derived from stored raw totals in evaluation JSONL files.
No re-tokenization or dataset access required.

Derived metrics and their prerequisites
----------------------------------------
Metric                       Prerequisites
-----------                  ---------------------------------
fertility                    total_tokens, total_words
compression_bytes            total_bytes, total_tokens
compression_chars            total_chars, total_tokens
reversible_ratio             reversible_docs_count, total_docs
unk_ratio                    unk_tokens_count, total_tokens
byte_fidelity                byte_edit_distance_total, byte_edit_denominator_total
effective_bytes_per_token    compression_bytes, byte_fidelity
ebpb                         total_tokens, total_bytes, byte_edit_distance_total, freq_cardinality
ebpb_full                    total_tokens, total_bytes, byte_edit_distance_total, vocab_size
char_fidelity                char_edit_distance_total, char_edit_denominator_total
effective_chars_per_token    compression_chars, char_fidelity
ebpc                         vocab_size, effective_chars_per_token
freq_cardinality_ratio       freq_cardinality, vocab_size
freq_90pct_mass_ratio        freq_90pct_mass_count, vocab_size

Rows that do not have all prerequisites for a given metric are left unchanged for
that metric, so this script is safe to run on partial or mixed-vintage files.
"""

import argparse
import json
import math
import traceback
from pathlib import Path

from tqdm import tqdm

from env import DEFAULT_MODEL_EVALUATIONS_PATH, HF_REPO_RESULTS
from tokenizer_evaluate import extract_result_zips


# ---------------------------------------------------------------------------
# Per-metric derivation functions (pure, no tokenizer dependency)
# ---------------------------------------------------------------------------

def _pos(v):
    """Return v if positive and non-None, else None."""
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _fmt(v, precision=3):
    if v is None:
        return None
    return round(max(float(v), 0), precision)


def _derive_fertility(row):
    t = _pos(row.get('total_tokens'))
    w = _pos(row.get('total_words'))
    if t is None or w is None:
        return None
    return _fmt(t / w)


def _derive_compression_bytes(row):
    b = _pos(row.get('total_bytes'))
    t = _pos(row.get('total_tokens'))
    if b is None or t is None:
        return None
    return _fmt(b / t)


def _derive_compression_chars(row):
    c = _pos(row.get('total_chars'))
    t = _pos(row.get('total_tokens'))
    if c is None or t is None:
        return None
    return _fmt(c / t)


def _derive_reversible_ratio(row):
    rd = row.get('reversible_docs_count')
    td = _pos(row.get('total_docs'))
    if rd is None or td is None:
        return None
    try:
        return _fmt(float(rd) / td, 4)
    except (TypeError, ValueError):
        return None


def _derive_unk_ratio(row):
    uc = row.get('unk_tokens_count')
    t = _pos(row.get('total_tokens'))
    if uc is None or t is None:
        return None
    try:
        return _fmt(float(uc) / t, 5)
    except (TypeError, ValueError):
        return None


def _derive_byte_fidelity(row):
    d = row.get('byte_edit_distance_total')
    n = _pos(row.get('byte_edit_denominator_total'))
    if d is None or n is None:
        return None
    try:
        return round(max(1.0 - float(d) / n, 0.0), 5)
    except (TypeError, ValueError):
        return None


def _derive_effective_bytes_per_token(row, byte_fidelity=None):
    cb = row.get('compression_bytes')
    if cb is None:
        cb = _derive_compression_bytes(row)
    if byte_fidelity is None:
        byte_fidelity = _derive_byte_fidelity(row)
    if cb is None or byte_fidelity is None:
        return None
    try:
        return _fmt(float(cb) * float(byte_fidelity))
    except (TypeError, ValueError):
        return None


def _derive_ebpb(row):
    t = _pos(row.get('total_tokens'))
    b = _pos(row.get('total_bytes'))
    d = row.get('byte_edit_distance_total')
    fc = row.get('freq_cardinality')
    if t is None or b is None or d is None or fc is None:
        return None
    try:
        fc = float(fc)
        if fc < 2:
            return None
        obs = max(fc, 2)
        value = (t * math.log2(obs) + 8 * float(d)) / b
        return _fmt(value)
    except (TypeError, ValueError):
        return None


def _derive_ebpb_full(row):
    t = _pos(row.get('total_tokens'))
    b = _pos(row.get('total_bytes'))
    d = row.get('byte_edit_distance_total')
    vs = row.get('vocab_size')
    if t is None or b is None or d is None or vs is None:
        return None
    try:
        vs = float(vs)
        if vs < 2:
            return None
        value = (t * math.log2(max(vs, 2)) + 8 * float(d)) / b
        return _fmt(value)
    except (TypeError, ValueError):
        return None


def _derive_char_fidelity(row):
    d = row.get('char_edit_distance_total')
    n = _pos(row.get('char_edit_denominator_total'))
    if d is None or n is None:
        return None
    try:
        return round(max(1.0 - float(d) / n, 0.0), 5)
    except (TypeError, ValueError):
        return None


def _derive_effective_chars_per_token(row, char_fidelity=None):
    cc = row.get('compression_chars')
    if cc is None:
        cc = _derive_compression_chars(row)
    if char_fidelity is None:
        char_fidelity = _derive_char_fidelity(row)
    if cc is None or char_fidelity is None:
        return None
    try:
        return _fmt(float(cc) * float(char_fidelity))
    except (TypeError, ValueError):
        return None


def _derive_ebpc(row, effective_chars_per_token=None):
    vs = _pos(row.get('vocab_size'))
    if effective_chars_per_token is None:
        effective_chars_per_token = _derive_effective_chars_per_token(row)
    if vs is None or effective_chars_per_token is None:
        return None
    try:
        ect = float(effective_chars_per_token)
        if ect <= 0:
            return None
        return _fmt(math.log2(max(vs, 2)) / ect)
    except (TypeError, ValueError):
        return None


def _derive_freq_cardinality_ratio(row):
    fc = row.get('freq_cardinality')
    vs = _pos(row.get('vocab_size'))
    if fc is None or vs is None:
        return None
    try:
        return round(float(fc) / vs, 5)
    except (TypeError, ValueError):
        return None


def _derive_freq_90pct_mass_ratio(row):
    mc = row.get('freq_90pct_mass_count')
    vs = _pos(row.get('vocab_size'))
    if mc is None or vs is None:
        return None
    try:
        return round(float(mc) / vs, 5)
    except (TypeError, ValueError):
        return None


# Ordered derivation plan: each entry is (output_col, derive_fn).
# Dependencies that cascade (e.g. ebpc depends on effective_chars_per_token) are
# handled inside the function by re-deriving if the stored value is missing.
DERIVATIONS = [
    ('fertility',                 _derive_fertility),
    ('compression_bytes',         _derive_compression_bytes),
    ('compression_chars',         _derive_compression_chars),
    ('reversible_ratio',          _derive_reversible_ratio),
    ('unk_ratio',                 _derive_unk_ratio),
    ('byte_fidelity',             _derive_byte_fidelity),
    ('effective_bytes_per_token', _derive_effective_bytes_per_token),
    ('ebpb',                      _derive_ebpb),
    ('ebpb_full',                 _derive_ebpb_full),
    ('char_fidelity',             _derive_char_fidelity),
    ('effective_chars_per_token', _derive_effective_chars_per_token),
    ('ebpc',                      _derive_ebpc),
    ('freq_cardinality_ratio',    _derive_freq_cardinality_ratio),
    ('freq_90pct_mass_ratio',     _derive_freq_90pct_mass_ratio),
]


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def _update_row(row: dict, force: bool) -> tuple[dict, list[str]]:
    """Derive and patch all metrics that have their prerequisites available.

    Returns (updated_row, list_of_changed_fields).
    """
    changed = []
    updated = dict(row)

    for col, fn in DERIVATIONS:
        existing = updated.get(col)
        # Skip if already present and not forcing
        if not force and existing is not None:
            continue
        new_val = fn(updated)
        if new_val is None:
            continue
        if existing != new_val:
            updated[col] = new_val
            changed.append(col)

    return updated, changed


def process_file(file_path: Path, force: bool) -> tuple[bool, list[str]]:
    """Process one JSONL evaluation file. Returns (was_modified, log_lines)."""
    logs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if not rows:
            logs.append(f"  {file_path.name}: empty, skipping.")
            return False, logs

        new_rows = []
        file_changed_cols: set[str] = set()
        for row in rows:
            updated_row, changed = _update_row(row, force)
            new_rows.append(updated_row)
            file_changed_cols.update(changed)

        if not file_changed_cols:
            return False, logs

        with open(file_path, 'w', encoding='utf-8') as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

        logs.append(f"  {file_path.name}: updated {sorted(file_changed_cols)}")
        return True, logs

    except Exception:
        logs.append(f"  ERROR {file_path.name}:\n{traceback.format_exc()}")
        return False, logs


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def recalculate(
    evaluations_dir: Path,
    upload_to_hub: bool = False,
    force: bool = False,
):
    if not evaluations_dir.exists():
        print(f"Evaluations directory not found: {evaluations_dir}")
        return

    result_files = sorted(evaluations_dir.glob('results_*.jsonl'))
    if not result_files:
        print(f"No result files found in {evaluations_dir}")
        return

    file_to_zip = extract_result_zips(evaluations_dir)

    print(f"Processing {len(result_files)} files (force={force}) ...")
    modified_files: list[Path] = []

    with tqdm(total=len(result_files), desc="Recalculating metrics") as pbar:
        for fp in result_files:
            was_modified, logs = process_file(fp, force)
            for line in logs:
                tqdm.write(line)
            if was_modified:
                modified_files.append(fp)
            pbar.update()

    print(f"\n{len(modified_files)} file(s) updated.")

    if upload_to_hub and modified_files:
        from huggingface_hub import CommitOperationAdd, HfApi
        from tokenizer_evaluate import update_files_in_zip

        zip_updates: dict[Path, dict[str, Path]] = {}
        loose_ops = []
        for fp in modified_files:
            fname = fp.name
            if fname in file_to_zip:
                zip_path = file_to_zip[fname]
                zip_updates.setdefault(zip_path, {})[fname] = fp
            else:
                loose_ops.append(
                    CommitOperationAdd(
                        path_in_repo=f"{DEFAULT_MODEL_EVALUATIONS_PATH}/{fname}",
                        path_or_fileobj=str(fp),
                    )
                )

        zip_ops = []
        for zip_path, members in zip_updates.items():
            print(f"Re-packing {zip_path.name} ({len(members)} file(s))...")
            update_files_in_zip(zip_path, members)
            zip_ops.append(
                CommitOperationAdd(
                    path_in_repo=f"{DEFAULT_MODEL_EVALUATIONS_PATH}/{zip_path.name}",
                    path_or_fileobj=str(zip_path),
                )
            )

        all_ops = loose_ops + zip_ops
        api = HfApi()
        print(f"Uploading {len(zip_ops)} zip(s) and {len(loose_ops)} loose file(s) ...")
        try:
            api.create_commit(
                repo_id=HF_REPO_RESULTS,
                operations=all_ops,
                commit_message="chore: Recalculate derived metrics",
                repo_type="dataset",
            )
            print("Upload complete.")
        except Exception as e:
            print(f"Upload error: {e}")

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate all derived evaluation metrics from stored raw totals. "
            "No re-tokenization required."
        )
    )
    parser.add_argument(
        '--evaluations_dir',
        type=Path,
        default=None,
        help="Path to the evaluations directory containing results_*.jsonl files. "
             "Defaults to the standard local path derived from env settings.",
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help="Upload modified files to Hugging Face Hub after recalculation.",
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Recalculate and overwrite even if the metric already exists in the row.",
    )
    args = parser.parse_args()

    if args.evaluations_dir is None:
        from tokenizer_evaluate import download_repos
        _, results_dir = download_repos()
        evaluations_dir = results_dir / DEFAULT_MODEL_EVALUATIONS_PATH
    else:
        evaluations_dir = args.evaluations_dir

    recalculate(
        evaluations_dir=evaluations_dir,
        upload_to_hub=args.upload,
        force=args.force,
    )
