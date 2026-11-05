import os
#os.environ["HF_HOME"] = os.path.abspath("data/tokenizers")
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, CommitOperationAdd
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import Tuple, Optional
import traceback
from collections import defaultdict
import multiprocessing

from env import HF_REPO_RESULTS, DATA_DIR
from tokenizer_evaluate import get_tokenizer_type, calculate_vocab_hash, get_tokenizer_config, get_tokenizer_hash_field, extract_result_zips, update_files_in_zip

# Define the paths relative to the script location
DEFAULT_RESULTS_PATH = "results"
DEFAULT_MODEL_EVALUATIONS_PATH = "evaluations"

def download_repos(data_dir=DATA_DIR):
    """
    Download the results repositories if not already present.

    Returns:
        (results_dir, file_to_zip) where file_to_zip maps filename -> source zip path.
    """
    results_dir = Path(data_dir) / DEFAULT_RESULTS_PATH
    
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

    file_to_zip = extract_result_zips(results_dir / DEFAULT_MODEL_EVALUATIONS_PATH)
    
    return results_dir, file_to_zip

def process_vocab_group(args):
    """
    Worker function to process a single group of models with the same hash.
    This function is designed to be called by a multiprocessing pool.
    """
    hash_value, files_info, upload_to_hub, evaluations_path, recalculate_hashes = args
    logs = []
    local_operations = []

    if not files_info:
        return local_operations, logs

    # Find the model with the most likes to act as the representative
    best_model_info = max(files_info, key=lambda x: x.get('likes', 0))
    model_to_load = best_model_info['model']

    # Get tokenizer type and optionally recalculate hashes for the entire group
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_to_load, trust_remote_code=True)
        algorithm, implementation = get_tokenizer_type(tokenizer)
        
        # Calculate hashes if requested
        new_vocab_hash = None
        new_tokenizer_config_hash = None
        if recalculate_hashes:
            new_vocab_hash, _ = calculate_vocab_hash(tokenizer)
            _, new_tokenizer_config_hash = get_tokenizer_config(tokenizer, new_vocab_hash)
            logs.append(f"Processing model '{model_to_load}'. Type: {algorithm}/{implementation}. New hashes: vocab={new_vocab_hash[:7]}, config={new_tokenizer_config_hash[:7]}")
        else:
            logs.append(f"Processing model '{model_to_load}'. Type: {algorithm}/{implementation}")
    except Exception as e:
        logs.append(f"Could not load tokenizer for '{model_to_load}'. Skipping. Error: {e}")
        return local_operations, logs

    # Update all files in this group
    for file_info in files_info:
        file_path = file_info['file_path']
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                continue

            updated_lines = []
            has_changed = False
            for line in lines:
                data = json.loads(line)
                
                # Update algorithm and implementation
                if data.get('tokenizer_algorithm') != algorithm or data.get('tokenizer_implementation') != implementation:
                    has_changed = True
                data['tokenizer_algorithm'] = algorithm
                data['tokenizer_implementation'] = implementation
                
                # Update hashes if recalculating
                if recalculate_hashes:
                    if new_vocab_hash and data.get('vocab_hash') != new_vocab_hash:
                        has_changed = True
                        data['vocab_hash'] = new_vocab_hash
                    if new_tokenizer_config_hash and data.get('tokenizer_config_hash') != new_tokenizer_config_hash:
                        has_changed = True
                        data['tokenizer_config_hash'] = new_tokenizer_config_hash
                
                updated_lines.append(json.dumps(data, ensure_ascii=False))

            if has_changed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for line in updated_lines:
                        f.write(line + '\n')

                if upload_to_hub:
                    path_in_repo = f"{evaluations_path}/{file_path.name}"
                    local_operations.append(
                        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(file_path))
                    )
                logs.append(f"  - Updated {file_path.name}")

        except json.JSONDecodeError:
            logs.append(f"Skipping {file_path.name}: Invalid JSON.")
        except Exception:
            logs.append(f"An unexpected error occurred with {file_path.name}:\n{traceback.format_exc()}")
    
    return local_operations, logs

def fix_tokenizer_types_for_all_models(upload_to_hub: bool = True, recalculate_hashes: bool = False):
    """
    Iterates through all evaluation result files, determines the correct
    tokenizer algorithm and implementation, and optionally recalculates hashes.
    This version is optimized to group by the appropriate hash field and uses multiprocessing.
    """
    # Download data first
    print("Ensuring data is up-to-date...")
    results_dir, file_to_zip = download_repos()

    # Initialize HF API
    api = HfApi()

    # Define the directory containing the evaluation files
    evaluations_dir = results_dir / DEFAULT_MODEL_EVALUATIONS_PATH

    if not evaluations_dir.exists():
        print(f"Error: Evaluations directory not found at '{evaluations_dir}'")
        return

    # Get all result files
    result_files = list(evaluations_dir.glob("results_*.jsonl"))

    if not result_files:
        print(f"No result files found in '{evaluations_dir}'")
        return

    # Step 1: Prepare individual files for processing
    print("Processing all models individually...")
    individual_files = []
    
    for file_path in tqdm(result_files, desc="Scanning files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if not first_line:
                    continue
                data = json.loads(first_line)
                model_name = data.get('model')
                likes = data.get('likes', 0)

                if model_name:
                    individual_files.append({
                        'file_path': file_path,
                        'model': model_name,
                        'likes': likes,
                    })
                else:
                    tqdm.write(f"Skipping {file_path.name}: missing 'model'.")
        except (json.JSONDecodeError, IOError) as e:
            tqdm.write(f"Skipping {file_path.name} due to error: {e}")

    print(f"Found {len(result_files)} model evaluation files to update.")
    print(f"Processing {len(individual_files)} models individually.")
    
    if recalculate_hashes:
        print("Hash recalculation is enabled - will update vocab_hash and tokenizer_config_hash fields.")

    per_file_operations = []

    # Step 2: Process each file individually in parallel
    worker_args = [
        (f"file_{i}", [file_info], upload_to_hub, DEFAULT_MODEL_EVALUATIONS_PATH, recalculate_hashes)
        for i, file_info in enumerate(individual_files)
    ]

    with multiprocessing.Pool(processes=60) as pool:
        task_desc = "Processing models (fixing types and hashes)" if recalculate_hashes else "Processing models (fixing types)"
        with tqdm(total=len(worker_args), desc=task_desc) as pbar:
            for file_operations, file_logs in pool.imap_unordered(process_vocab_group, worker_args):
                for log in file_logs:
                    tqdm.write(log)
                if file_operations:
                    per_file_operations.extend(file_operations)
                pbar.update()

    # Post-process operations: group files-from-zips by their source zip
    zip_updates: dict = {}   # zip_path -> {member_name: local_file_path}
    loose_operations = []
    for op in per_file_operations:
        fname = Path(op.path_in_repo).name
        if fname in file_to_zip:
            zip_path = file_to_zip[fname]
            local_path = Path(op.path_or_fileobj)
            zip_updates.setdefault(zip_path, {})[fname] = local_path
        else:
            loose_operations.append(op)

    # Rebuild affected zips
    zip_operations = []
    for zip_path, updated_members in zip_updates.items():
        print(f"Re-packing {zip_path.name} with {len(updated_members)} updated file(s)...")
        update_files_in_zip(zip_path, updated_members)
        zip_operations.append(
            CommitOperationAdd(
                path_in_repo=f"{DEFAULT_MODEL_EVALUATIONS_PATH}/{zip_path.name}",
                path_or_fileobj=str(zip_path),
            )
        )

    all_operations = loose_operations + zip_operations

    # After loop, if there are operations, create a single commit
    if upload_to_hub and all_operations:
        print(f"\nUploading {len(zip_operations)} zip(s) and {len(loose_operations)} loose file(s) to the Hub...")
        try:
            commit_message = "chore: Fix tokenizer algorithm, implementation and hashes" if recalculate_hashes else "chore: Fix tokenizer algorithm and implementation"
            api.create_commit(
                repo_id=HF_REPO_RESULTS,
                operations=all_operations,
                commit_message=commit_message,
                repo_type="dataset",
            )
            print("Successfully uploaded all changes.")
        except Exception as e:
            print(f"An error occurred during upload: {e}")
    elif upload_to_hub:
        print("\nNo files needed updating, so no upload was performed.")

    print("\nUpdate complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fix tokenizer algorithm, implementation and optionally recalculate hashes for all evaluated models.')
    parser.add_argument('--not_upload', action='store_true', help='Do not upload updated files to Hugging Face Hub.')
    parser.add_argument('--recalculate_hashes', action='store_true', help='Recalculate vocab_hash and tokenizer_config_hash fields.')
    args = parser.parse_args()

    fix_tokenizer_types_for_all_models(upload_to_hub=not args.not_upload, recalculate_hashes=args.recalculate_hashes) 