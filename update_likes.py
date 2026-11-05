import os
import json
import time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download, CommitOperationAdd
from huggingface_hub.errors import HfHubHTTPError
from tqdm import tqdm
import argparse

from env import HF_REPO_RESULTS, HF_REPO_BENCHMARK, DATA_DIR
from tokenizer_evaluate import extract_result_zips, update_files_in_zip, get_author_info

# Define the paths relative to the script location
DEFAULT_RESULTS_PATH = "results"
DEFAULT_BENCHMARK_PATH = "benchmark"
DEFAULT_MODEL_EVALUATIONS_PATH = "evaluations"


def _call_with_retry(fn, *args, max_retries=10, default_wait=60, **kwargs):
    """Call fn(*args, **kwargs), retrying on HTTP 429 rate-limit responses."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = int(e.response.headers.get("Retry-After", default_wait))
                wait = max(retry_after, 1)
                tqdm.write(f"Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise
    # Final attempt — let exception propagate
    return fn(*args, **kwargs)

def download_repos(data_dir=DATA_DIR):
    """
    Download the benchmark and results repositories if not already present.

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

def update_model_metadata_for_all_models(upload_to_hub: bool = True):
    """
    Iterates through all evaluation result files, fetches the latest
    model metadata (likes, sha, created_at, downloads_all_time) from the 
    Hugging Face Hub, and updates the files.
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

    print(f"Found {len(result_files)} model evaluation files to update.")

    # Track loose-file operations and per-zip updates separately
    loose_operations = []
    zip_updates: dict = {}   # zip_path -> {member_name: local_file_path}

    # Cache author info to avoid redundant API calls for models from the same author
    author_info_cache: dict = {}

    # Loop through each file with a progress bar
    for file_path in tqdm(result_files, desc="Updating model metadata"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                continue

            # Get model name from the first line
            first_line_data = json.loads(lines[0])
            model_name = first_line_data.get('model')

            if not model_name:
                tqdm.write(f"Skipping {file_path.name}: 'model' key not found.")
                continue

            # Get model metadata from the Hub
            try:
                model_info_data = _call_with_retry(
                    api.model_info, model_name, expand=['sha', 'likes', 'downloadsAllTime', 'createdAt']
                )
                likes = model_info_data.likes
                sha = getattr(model_info_data, 'sha', None)
                created_at = getattr(model_info_data, 'created_at', None)
                downloads_all_time = getattr(model_info_data, 'downloads_all_time', 0)
                
                # Convert created_at to ISO format if needed
                if created_at is not None and not isinstance(created_at, str):
                    created_at = created_at.isoformat()
                    
            except Exception as e:
                # If model is not found or another error occurs, keep the old values or use defaults
                likes = first_line_data.get('likes', 0)
                sha = first_line_data.get('sha', None)
                created_at = first_line_data.get('created_at', None)
                downloads_all_time = first_line_data.get('downloads_all_time', 0)
                tqdm.write(f"Could not fetch metadata for '{model_name}': {e}. Using existing values.")

            # Fetch author info (with per-author caching)
            author_info = _call_with_retry(get_author_info, api, model_name, cache=author_info_cache)

            # Update all lines with the new metadata
            has_changed = False
            if (first_line_data.get('likes', 0) != likes or 
                first_line_data.get('sha', None) != sha or 
                first_line_data.get('created_at', None) != created_at or 
                first_line_data.get('downloads_all_time', 0) != downloads_all_time or
                first_line_data.get('author_type') != author_info.get('author_type') or
                first_line_data.get('author_fullname') != author_info.get('author_fullname') or
                first_line_data.get('author_followers', 0) != author_info.get('author_followers', 0) or
                first_line_data.get('author_models', 0) != author_info.get('author_models', 0) or
                first_line_data.get('author_is_verified', False) != author_info.get('author_is_verified', False)):
                has_changed = True
            
            if has_changed:
                updated_lines = []

                for line in lines:
                    data = json.loads(line)

                    # Update all metadata fields
                    data['likes'] = likes
                    data['sha'] = sha
                    data['created_at'] = created_at
                    data['downloads_all_time'] = downloads_all_time
                    data['author_type'] = author_info.get('author_type')
                    data['author_fullname'] = author_info.get('author_fullname')
                    data['author_followers'] = author_info.get('author_followers', 0)
                    data['author_models'] = author_info.get('author_models', 0)
                    data['author_is_verified'] = author_info.get('author_is_verified', False)
                    
                    # Use ensure_ascii=False to handle non-ASCII characters gracefully
                    updated_lines.append(json.dumps(data, ensure_ascii=False))

                with open(file_path, 'w', encoding='utf-8') as f:
                    for line in updated_lines:
                        f.write(line + '\n')

                if upload_to_hub:
                    if file_path.name in file_to_zip:
                        # File lives inside a batch zip — collect for zip update
                        zip_path = file_to_zip[file_path.name]
                        zip_updates.setdefault(zip_path, {})[file_path.name] = file_path
                    else:
                        # Loose file — upload directly
                        loose_operations.append(
                            CommitOperationAdd(
                                path_in_repo=f"{DEFAULT_MODEL_EVALUATIONS_PATH}/{file_path.name}",
                                path_or_fileobj=str(file_path),
                            )
                        )
                    
                tqdm.write(f"Updated {file_path.name}")

        except json.JSONDecodeError:
            tqdm.write(f"Skipping {file_path.name}: Invalid JSON.")
        except Exception as e:
            tqdm.write(f"An unexpected error occurred with {file_path.name}: {e}")

    # Rebuild affected zips and add upload operations for them
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
            api.create_commit(
                repo_id=HF_REPO_RESULTS,
                operations=all_operations,
                commit_message="chore: Update model metadata (likes, sha, created_at, downloads_all_time, author fields)",
                repo_type="dataset",
            )
            print("Successfully uploaded all changes.")
        except Exception as e:
            print(f"An error occurred during upload: {e}")
    elif upload_to_hub:
        print("\nNo files needed updating, so no upload was performed.")

    print("\nUpdate complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update model metadata (likes, sha, created_at, downloads_all_time) for all evaluated models.')
    parser.add_argument('--not_upload', action='store_true', help='Do not upload updated files to Hugging Face Hub.')
    args = parser.parse_args()

    update_model_metadata_for_all_models(upload_to_hub=not args.not_upload) 