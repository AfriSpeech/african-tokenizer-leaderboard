import os
from transformers import AutoTokenizer, PreTrainedTokenizer
import pandas as pd
from pathlib import Path
import shutil
import zipfile
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import yaml
import json
import gzip
import hashlib
import datetime
import numpy as np
import math
import importlib
from collections import defaultdict, Counter
from typing import Tuple, Dict, List, Optional
from copy import deepcopy
import re
import traceback
import unicodedata
import random
import time
from scipy.integrate import simpson
from tqdm.auto import tqdm

from env import (
    HF_REPO_RESULTS,
    HF_REPO_BENCHMARK,
    DATA_DIR,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_RESULTS_PATH,
    DEFAULT_MODEL_EVALUATIONS_PATH,
    DEFAULT_TOKENIZER_METADATA_PATH
)

from util import (
    ALL_COLUMNS,
    PARITY_COLUMNS,
    ensure_dataframe_has_columns,
)

DUPLICATE_MAIN_FIELD = "duplicates_plus_lower" # can be "duplicates", "duplicates_plus_lower" or "duplicates_plus_lower_normalized", "duplicates_plus_lower_normalized_stripped"
DEFAULT_BATCH_SIZE = 10000
TOKENIZER_HASH_FIELD = 'tokenizer_config_hash' # it was vocab_hash, but we now use tokenizer_config_hash
BYTE_FIDELITY_EXACT_MAX_OPS = 1024*1024*32*32  # max(orig_bytes * decoded_bytes) before chunking
BYTE_FIDELITY_CHUNK_BYTES = 1024*16
CHAR_FIDELITY_EXACT_MAX_OPS = 1024*1024*32*32
CHAR_FIDELITY_CHUNK_CHARS = 1024*16

# Note: Cache-related constants and ENABLE_LOAD_CACHE moved to results_aggregator.py

# Initialize HF API
api = HfApi()

# Helper function to get the current tokenizer hash field name
def get_tokenizer_hash_field():
    """Get the current field name used for tokenizer hash-based duplicate detection."""
    return TOKENIZER_HASH_FIELD

_LEVENSHTEIN_DISTANCE = None

def get_levenshtein_distance():
    """Lazily import the rapidfuzz Levenshtein distance implementation."""
    global _LEVENSHTEIN_DISTANCE
    if _LEVENSHTEIN_DISTANCE is None:
        levenshtein_module = importlib.import_module("rapidfuzz.distance.Levenshtein")
        _LEVENSHTEIN_DISTANCE = levenshtein_module.distance
    return _LEVENSHTEIN_DISTANCE

def generate_model_key(model_name: str, revision: str = "main", subfolder: str = None, _target_mode: bool = False) -> str:
    """
    Generate a standardized model key from model name, revision, optional subfolder, and target mode.
    Sanitizes text and applies size limits for consistency with filename generation.
    
    Args:
        model_name: The model name
        revision: The model revision (default: 'main')
        subfolder: Optional subfolder within the model repository
        _target_mode: Whether target mode is enabled (default: False)
        
    Returns:
        A sanitized model key string in the format: model_name_revision[_subfolder][__target_mode]
    """
    # Sanitize and apply size limits consistent with filename generation
    safe_model_name = _sanitize_text(model_name)[-120:]  # Last 120 characters
    safe_revision = _sanitize_text(revision)[:40]        # First 40 characters
    
    model_key = f"{safe_model_name}_{safe_revision}"
    if subfolder:
        safe_subfolder = _sanitize_text(subfolder)[:40]  # First 40 characters
        model_key = f"{safe_model_name}_{safe_revision}_{safe_subfolder}"
    
    if _target_mode:
        model_key = f"{model_key}__target_mode"
    
    return model_key

# Load dataset metadata
def load_dataset_meta(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = yaml.safe_load(f)
    return meta

# Generator to yield lines from a .jsonl.gz file
def yield_jsonl_gz(filepath):
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

# Calculate hash for an individual file
def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


def get_filtered_vocab_and_remove_special_tokens(tokenizer: PreTrainedTokenizer):
    if hasattr(tokenizer, 'get_vocab'):
        _vocab = tokenizer.get_vocab()
    elif hasattr(tokenizer, 'vocab'):
        vocab = tokenizer.vocab
    else:
        raise ValueError("Tokenizer does not have a vocab")
    
    vocab = {}
    for k, v in _vocab.items():
        if isinstance(k, str):
            vocab[k] = v
        else:
            vocab[k.decode('utf-8', errors='backslashreplace')] = v

    special_tokens_to_exclude = set()

    # Method 1: Using all_special_ids to get token strings from the main vocab
    if hasattr(tokenizer, 'all_special_ids'):
        special_token_ids = set(tokenizer.all_special_ids)
        for token_str, token_id in vocab.items():
            if token_id in special_token_ids:
                special_tokens_to_exclude.add(token_str)
                #print(f"Special token all_special_ids: {token_str}")

    # Method 2: Using special_tokens_map (covers bos, eos, pad, unk, mask, and often additional_special_tokens)
    if hasattr(tokenizer, 'special_tokens_map'):
        for key, value in tokenizer.special_tokens_map.items():
            if isinstance(value, str):
                special_tokens_to_exclude.add(value)
                #print(f"Special token special_tokens_map: {value}")
            elif isinstance(value, list):  # Handles lists like additional_special_tokens
                for token_str in value:
                    if isinstance(token_str, str):
                        special_tokens_to_exclude.add(token_str)
                        #print(f"Special token special_tokens_map list: {token_str}")

    # Method 3: Using added_tokens_decoder for tokens added via add_tokens/add_special_tokens
    if hasattr(tokenizer, 'added_tokens_decoder'):
        for token_id, added_token_obj in tokenizer.added_tokens_decoder.items():
            # added_token_obj is usually an AddedToken object
            token = added_token_obj
            if hasattr(added_token_obj, 'content'):
                token = added_token_obj.content
            if token.strip():
                special_tokens_to_exclude.add(token)
                #print(f"Special token added_tokens_decoder: {token}")
            else:
                #print(f"Skipping empty added_tokens_decoder token id {token_id}: ' 'x{len(token)}")
                pass
    
    # Incluide spacing formating special tokens
    include_in_filtered = [s for s in special_tokens_to_exclude if s.strip() != s]
    for s in include_in_filtered:
        special_tokens_to_exclude.remove(s) # remove spacing special tokens

    # Filter out special tokens from vocab
    _filtered_vocab = {k: v for k, v in vocab.items() if k not in special_tokens_to_exclude}

    # Remove bytes from vocab, convert to string
    filtered_vocab = {}
    for k, v in _filtered_vocab.items():
        if isinstance(k, bytes):
            token_str = f"b_{k.hex()}"
            try:
                token_str.encode('utf-8')
            except UnicodeEncodeError:
                pass

            filtered_vocab[token_str] = v
        else:
            filtered_vocab[k] = v
    return filtered_vocab

# Calculate hash for a tokenizer
def calculate_vocab_hash(tokenizer: PreTrainedTokenizer) -> Tuple[str, Optional[dict]]:
    """Calculate a hash to uniquely identify the tokenizer. And return the filtered vocab."""
    # Try to use vocab for hashing
    # If possible, remove all special tokens to calculate a more accurate hashing
    try:
        filtered_vocab = get_filtered_vocab_and_remove_special_tokens(tokenizer)
        vocab_str = json.dumps(filtered_vocab, sort_keys=True)
        return hashlib.md5(vocab_str.encode('utf-8')).hexdigest(), filtered_vocab
    except ValueError as e:
        # Fallback to tokenizer config
        config = tokenizer.init_kwargs if hasattr(tokenizer, 'init_kwargs') else {'name_or_path': tokenizer.name_or_path}
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest(), None

# added </w> and @@ later, they are actually sufixes, but i am too lazy to rename this now...
POSSIBLE_PREFIXES = ['Ġ', chr(9601), "##", "</w>", "@@"]
def detect_vocab_prefix(vocab: Dict[str, int], possible_prefixes: List[str] = POSSIBLE_PREFIXES):

    _vocab = {}
    for k, v in vocab.items():
        if isinstance(k, str):
            _vocab[k] = v
        else:
            _vocab[k.decode('utf-8', errors='backslashreplace')] = v

    prefixes = []
    for token_str in _vocab.keys():
        for prefix in POSSIBLE_PREFIXES:
            if prefix in token_str:
                prefixes.append(prefix)

    counts = Counter(prefixes)
    
    candidate = POSSIBLE_PREFIXES[0]
    #get biggest possible prefix count
    for prefix in POSSIBLE_PREFIXES[1:]:
        if counts[prefix] > counts[candidate]:
            candidate = prefix

    #Works for most vocabs without prefix
    if counts[candidate] < int(len(_vocab) * 0.03):
        return None

    return candidate

def _normalize_for_nfkd_mn(text_token):
    """Helper to normalize text using NFKD and strip Mn characters."""
    if not text_token: # handle empty string if it somehow gets here
        return ""
    normalized = unicodedata.normalize('NFKD', text_token)
    stripped = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    return stripped

def detect_near_duplicated_tokens(vocab_dict: dict, detect_prefix: bool = True, vocab_prefix: str | None = None, verbose: bool = False):
    """
    Analyzes a tokenizer's vocabulary for duplicates based on capitalization,
    leading prefixes, and multi-digit numbers.

    Args:
        vocab_dict: The vocabulary dictionary of the tokenizer.
        detect_prefix (bool): If True, the function will try to detect the prefix.
        vocab_prefix (str): Force an specific prefix used by this tokenizer (e.g., "Ġ" or chr(9601)).
                            Can be None. Only used if detect_prefix is False.
        verbose (bool): If True, the function will print debug information.
    Returns:
        dict: A dictionary with percentages and counts of duplicates.
    """
    vocab_keys = sorted(list(vocab_dict.keys()))
    n_vocab = len(vocab_keys)

    if detect_prefix:
        vocab_prefix = detect_vocab_prefix(vocab_dict)
        
    ### Count different types of duplicates for analysis, each step with increasing normalization ###

    # S1: Space only duplicates e.g. [' ', ' \n', '\t\t ', 'Ġ\t\t ']
    s1_duplicate_space_set = set()

    # S2: Remove prefix duplicates e.g. 'Ola' -> ['ĠOla', 'Ola'] (after S1)
    s2_duplicate_rprefix_mapping = defaultdict(list)

    # S3: Remove prefix + Strip space duplicates e.g. '{' -> ['\t\t{', '{ '] (after S2)
    s3_duplicate_rprefix_strip_mapping = defaultdict(list)

    # S4: Remove prefix + Strip space + Number only duplicates (consider only the first number) e.g. 1 -> ['12', '121', 'Ġ155'] (after S3)
    s4_duplicate_rprefix_strip_digit_mapping = defaultdict(list)

    # S5: Remove prefix + Strip space + uncapitalized duplicates e.g. 'danger' -> ['danger', 'Ġdanger', 'Danger', 'GDANGER'] (after S4)
    s5_duplicate_rprefix_strip_lower_mapping = defaultdict(list)

    # S6: Remove prefix + Strip space + uncapitalized + normalized duplicates e.g. 'olá' -> ['ola', 'Olá', 'Ġôla'] (after S5)
    s6_duplicate_rprefix_strip_lower_normalized_mapping = defaultdict(list)
    
    digits = ['0','1','2','3','4','5','6','7','8','9']

    # --- Identify duplicates for each category ---
    for token in vocab_keys:
        original_token_str = str(token)

        # prefix removal
        token_rprefix = original_token_str

        if vocab_prefix in ['</w>', '@@'] and original_token_str.endswith(vocab_prefix):
                token_rprefix = original_token_str[:-len(vocab_prefix)]
        elif vocab_prefix and original_token_str.startswith(vocab_prefix):
            token_rprefix = original_token_str[len(vocab_prefix):]
        
        # Process for S1 (Empty Space-only)
        # If after prefix removal, it becomes empty (e.g. token was just "Ġ"), treat as space for S1 count logic.
        # This ensures tokens like "Ġ" are handled consistently if they become effectively empty.
        if len(token_rprefix.strip()) == 0:
            s1_duplicate_space_set.add(original_token_str) # Add original token to space set
            continue # Skip empty space-only tokens for further specific processing stages

        # S2: Populate for removed prefix
        s2_duplicate_rprefix_mapping[token_rprefix].append(original_token_str)

        # S3: Populate for removed prefix + stripped
        token_rprefix_strip = token_rprefix.strip()
        s3_duplicate_rprefix_strip_mapping[token_rprefix_strip].append(original_token_str)

        # S4: Populate for removed prefix + stripped + digits (check only first digit)
        if token_rprefix_strip[0] in digits and token_rprefix_strip.isdigit():
            s4_duplicate_rprefix_strip_digit_mapping[token_rprefix_strip[0]].append(original_token_str)

        # S5: Populate for removed prefix + stripped + lowercased
        token_rprefix_strip_lower = token_rprefix_strip.lower()
        s5_duplicate_rprefix_strip_lower_mapping[token_rprefix_strip_lower].append(original_token_str)

        # S6: Populate for removed prefix + stripped + lowercased + normalized
        token_rprefix_strip_lower_normalized = _normalize_for_nfkd_mn(token_rprefix_strip_lower)
        s6_duplicate_rprefix_strip_lower_normalized_mapping[token_rprefix_strip_lower_normalized].append(original_token_str)

    #s2
    #'A': ['##A'] # 
    #'A\n': ['##A\n'] #

    #S3
    #'A': ['A\n', '##A\n\n']

    already_counted_as_duplicate = set()

    def count_duplicates(mapping):
        groups = {}
        count = 0
        for key, duplicates in mapping.items():
            if len(duplicates) > 1:
                canonical_token = None
                has_new_duplicated_token_in_this_key_group = False
                for i, token_val in enumerate(duplicates): # renamed 'token' to 'token_val' to avoid conflict
                    if token_val not in already_counted_as_duplicate:
                        if not canonical_token:
                            canonical_token = token_val
                        else:
                            already_counted_as_duplicate.add(token_val)
                            count += 1
                            has_new_duplicated_token_in_this_key_group = True
                if has_new_duplicated_token_in_this_key_group:
                    groups[canonical_token] = duplicates
        return count, groups

    # S1 Count:
    # For space duplicates, we count all unique space tokens first.
    # The "-1" means we are counting how many *extra* variations exist beyond one canonical form.
    # If only one space token exists (e.g. " "), count is 0. If {' ', '  '}, count is 1.
    s1_count_space_duplicates = 0
    s1_groups_space = {}
    if len(s1_duplicate_space_set) > 1 : # Only count if there's more than one variant
        duplicate_space_list = list(s1_duplicate_space_set)
        canonical_space = duplicate_space_list[0]
        s1_groups_space[canonical_space] = duplicate_space_list[1:]
        s1_count_space_duplicates = len(s1_duplicate_space_set) -1 # Number of *additional* duplicates
        already_counted_as_duplicate.update(s1_groups_space[canonical_space]) # Mark all space variants as "counted" for subsequent steps
    
    s2_count_rprefix_duplicates, s2_groups_rprefix = count_duplicates(s2_duplicate_rprefix_mapping)
    s3_count_rprefix_strip_duplicates, s3_groups_rprefix_strip = count_duplicates(s3_duplicate_rprefix_strip_mapping)
    s4_count_rprefix_strip_digit_duplicates, s4_groups_rprefix_strip_digit = count_duplicates(s4_duplicate_rprefix_strip_digit_mapping)
    s5_count_rprefix_strip_lower_duplicates, s5_groups_rprefix_strip_lower = count_duplicates(s5_duplicate_rprefix_strip_lower_mapping)
    s6_count_rprefix_strip_lower_normalized_duplicates, s6_groups_rprefix_strip_lower_normalized = count_duplicates(s6_duplicate_rprefix_strip_lower_normalized_mapping)
    
    # Total duplicates are the sum of disjoint counts from each step
    # total_duplicates includes S1 through S4
    total_duplicates = (s1_count_space_duplicates +
                                    s2_count_rprefix_duplicates +
                                    s3_count_rprefix_strip_duplicates +
                                    s4_count_rprefix_strip_digit_duplicates)
    # total_duplicates_lower includes S1 through S5
    total_duplicates_plus_lower = total_duplicates + s5_count_rprefix_strip_lower_duplicates
    # total_duplicates_with_norm includes S1 through S6
    total_duplicates_plus_lower_norm = total_duplicates_plus_lower + s6_count_rprefix_strip_lower_normalized_duplicates

    if verbose:
        # Samples for printing
        s1_pprint_sample_space = [f"{json.dumps(canonical, ensure_ascii=False)} -> {json.dumps(duplicates, ensure_ascii=False)}" for canonical, duplicates in s1_groups_space.items()]
        s2_pprint_sample_rprefix = [f"{json.dumps(canonical, ensure_ascii=False)} -> {json.dumps(duplicates, ensure_ascii=False)}" for canonical, duplicates in s2_groups_rprefix.items()]
        s3_pprint_sample_rprefix_strip = [f"{json.dumps(canonical, ensure_ascii=False)} -> {json.dumps(duplicates, ensure_ascii=False)}" for canonical, duplicates in s3_groups_rprefix_strip.items()]
        s4_pprint_sample_rprefix_strip_digit = [f"{json.dumps(canonical, ensure_ascii=False)} -> {json.dumps(duplicates, ensure_ascii=False)}" for canonical, duplicates in s4_groups_rprefix_strip_digit.items()]
        s5_pprint_sample_rprefix_strip_lower = [f"{json.dumps(canonical, ensure_ascii=False)} -> {json.dumps(duplicates, ensure_ascii=False)}" for canonical, duplicates in s5_groups_rprefix_strip_lower.items()]
        s6_pprint_sample_rprefix_strip_lower_normalized = [f"{json.dumps(canonical, ensure_ascii=False)} -> {json.dumps(duplicates, ensure_ascii=False)}" for canonical, duplicates in s6_groups_rprefix_strip_lower_normalized.items()]
    
        # Debugging counts
        print(f"\n--- Debugging (Near-duplicated vocab tokens analysis) ---")
        print(f"Vocab size: {n_vocab}")
        print(f"Prefix used for analysis: {vocab_prefix if vocab_prefix else 'None'}")
        print(f"S1. Found {s1_count_space_duplicates} space-only duplicates (variants beyond first). Sample: {'; '.join(random.sample(s1_pprint_sample_space, min(5, len(s1_pprint_sample_space))))}")
        print(f"S2. Found {s2_count_rprefix_duplicates} removed-prefix duplicates (not counted in S1). Sample: {'; '.join(random.sample(s2_pprint_sample_rprefix, min(5, len(s2_pprint_sample_rprefix))))}")
        print(f"S3. Found {s3_count_rprefix_strip_duplicates} stripped-space duplicates (post-S2, not in S1-S2). Sample: {'; '.join(random.sample(s3_pprint_sample_rprefix_strip, min(5, len(s3_pprint_sample_rprefix_strip))))}")
        print(f"S4. Found {s4_count_rprefix_strip_digit_duplicates} digits duplicates (post-S3, not in S1-S3). Sample: {'; '.join(random.sample(s4_pprint_sample_rprefix_strip_digit, min(1, len(s4_pprint_sample_rprefix_strip_digit))))}")
        print(f"S5. Found {s5_count_rprefix_strip_lower_duplicates} uncapitalized duplicates (post-S3 lowercased, not in S1-S4). Sample: {'; '.join(random.sample(s5_pprint_sample_rprefix_strip_lower, min(5, len(s5_pprint_sample_rprefix_strip_lower))))}")
        print(f"S6. Found {s6_count_rprefix_strip_lower_normalized_duplicates} normalized duplicates (post-S5 normalized, not in S1-S5). Sample: {'; '.join(random.sample(s6_pprint_sample_rprefix_strip_lower_normalized, min(5, len(s6_pprint_sample_rprefix_strip_lower_normalized))))}")
        print(f"Total duplicates (S1~S4): {total_duplicates}, Perc Duplicates: {total_duplicates*100 / n_vocab if n_vocab > 0 else 0:.2f}%")
        print(f"Total duplicates plus lower (S1~S5): {total_duplicates_plus_lower}, Perc Duplicates: {total_duplicates_plus_lower*100 / n_vocab if n_vocab > 0 else 0:.2f}%")
        print(f"Total duplicates plus lower and norm (S1~S6): {total_duplicates_plus_lower_norm}, Perc Duplicates: {total_duplicates_plus_lower_norm*100 / n_vocab if n_vocab > 0 else 0:.2f}%")

 
    results = {
        'n_vocab': n_vocab,
        'vocab_prefix': vocab_prefix,

        'count_by_step': {
            's1_num_space_duplicates': s1_count_space_duplicates,
            's2_num_rprefix_duplicates': s2_count_rprefix_duplicates,
            's3_num_rprefix_strip_duplicates': s3_count_rprefix_strip_duplicates,
            's4_num_rprefix_strip_digit_duplicates': s4_count_rprefix_strip_digit_duplicates,
            's5_num_rprefix_strip_lower_duplicates': s5_count_rprefix_strip_lower_duplicates,
            's6_num_rprefix_strip_lower_normalized_duplicates': s6_count_rprefix_strip_lower_normalized_duplicates,
        },

        'groups_by_step': {
            's1_groups_space_duplicates': s1_groups_space,
            's2_groups_rprefix_duplicates': s2_groups_rprefix,
            's3_groups_rprefix_strip_duplicates': s3_groups_rprefix_strip,
            's4_groups_rprefix_strip_digit_duplicates': s4_groups_rprefix_strip_digit,
            's5_groups_rprefix_strip_lower_duplicates': s5_groups_rprefix_strip_lower,
            's6_groups_rprefix_strip_lower_normalized_duplicates': s6_groups_rprefix_strip_lower_normalized,
        },
        
        'total_duplicates': total_duplicates, # Sum of S1-S4
        'total_duplicates_plus_lower': total_duplicates_plus_lower, # Sum of S1-S5
        'total_duplicates_plus_lower_norm': total_duplicates_plus_lower_norm, # Sum of S1-S6

        'duplicates': total_duplicates / n_vocab if n_vocab > 0 else 0,
        'duplicates_plus_lower': total_duplicates_plus_lower / n_vocab if n_vocab > 0 else 0,
        'duplicates_plus_lower_norm': total_duplicates_plus_lower_norm / n_vocab if n_vocab > 0 else 0,
    }
    return results

#default format
def default_format(value, precision=3):
    return round(value, precision) if value > 0 else 0

# Calculate fertility (tokens per word) with rounding
def calculate_fertility(tokens, words):
    """Calculate fertility (tokens per word) with rounding."""
    return default_format(tokens / words)

# Calculate compression_chars (characters per token) with rounding
def calculate_compression(chars, tokens):
    """Calculate compression_chars (characters per token) with rounding."""
    return default_format(chars / tokens)

# Calculate parity (tokens_a / tokens_b) with rounding
def calculate_parity(tokens_a, tokens_b):
    """Calculate parity (tokens_a / tokens_b) with rounding."""
    return default_format(tokens_a / tokens_b)

def calculate_reversible_ratio(reversible_docs_count, text_count):
    """Calculate reversible ratio (reversible_docs_count / text_count) with rounding."""
    return default_format(reversible_docs_count / text_count, 4)

def calculate_unk_ratio(count_unk_tokens, total_tokens):
    """Calculate unk ratio (count_unk_tokens / total_tokens) with rounding."""
    return default_format(count_unk_tokens / total_tokens, 5)

def _split_bytes_into_chunks(byte_string: bytes, chunk_count: int) -> List[bytes]:
    """Split a byte string into near-equal contiguous chunks."""
    if chunk_count <= 1:
        return [byte_string]

    boundaries = [
        round(index * len(byte_string) / chunk_count)
        for index in range(chunk_count + 1)
    ]
    return [
        byte_string[boundaries[index]:boundaries[index + 1]]
        for index in range(chunk_count)
    ]

def calculate_byte_edit_stats(original_text: str, decoded_text: str) -> Tuple[int, int]:
    """Return byte-level edit stats, chunking oversized documents to avoid quadratic blowups."""
    chunked = False
    original_bytes = original_text.encode('utf-8')
    decoded_bytes = decoded_text.encode('utf-8')
    max_length = max(len(original_bytes), len(decoded_bytes))

    if original_bytes == decoded_bytes:
        return 0, max(max_length, 1), chunked

    levenshtein_distance = get_levenshtein_distance()
    if len(original_bytes) * len(decoded_bytes) <= BYTE_FIDELITY_EXACT_MAX_OPS:
        distance = levenshtein_distance(original_bytes, decoded_bytes)
        return distance, max(max_length, 1), chunked
    
    chunked = True
    chunk_count = math.ceil(max_length / BYTE_FIDELITY_CHUNK_BYTES)
    original_chunks = _split_bytes_into_chunks(original_bytes, chunk_count)
    decoded_chunks = _split_bytes_into_chunks(decoded_bytes, chunk_count)

    total_distance = 0
    total_denominator = 0
    for original_chunk, decoded_chunk in zip(original_chunks, decoded_chunks):
        total_distance += levenshtein_distance(original_chunk, decoded_chunk)
        total_denominator += max(len(original_chunk), len(decoded_chunk), 1)

    return total_distance, total_denominator, chunked

def calculate_byte_fidelity(byte_edit_distance_total: int, byte_edit_denominator_total: int) -> float:
    """Calculate byte-level reconstruction fidelity from aggregated edit stats."""
    if byte_edit_denominator_total <= 0:
        return 0
    fidelity = 1 - (byte_edit_distance_total / byte_edit_denominator_total)
    return round(max(fidelity, 0), 5)

def calculate_effective_bytes_per_token(compression_bytes: float, byte_fidelity: float) -> float:
    """Calculate effective bytes/token after discounting lossy decoding."""
    return default_format(compression_bytes * byte_fidelity)

def calculate_ebpb(total_tokens, total_bytes, byte_edit_distance_total, freq_cardinality) -> float:
    """Calculate Effective Bits Per Byte (rate-distortion form). Lower is better.

    EBPB = (T * log2(V_obs) + 8 * D_byte) / B

    where T = total_tokens, V_obs = max(freq_cardinality, 2),
    D_byte = byte_edit_distance_total, B = total_bytes.
    Uses observed vocabulary cardinality — tokens actually seen in the corpus.
    """
    if not total_tokens or not total_bytes or not freq_cardinality or freq_cardinality < 2:
        return None
    obs = max(int(freq_cardinality), 2)
    value = (total_tokens * math.log2(obs) + 8 * byte_edit_distance_total) / total_bytes
    return default_format(max(value, 0))


def calculate_ebpb_full(total_tokens, total_bytes, byte_edit_distance_total, vocab_size) -> float:
    """Calculate Effective Bits Per Byte using full declared vocabulary. Lower is better.

    EBPB_full = (T * log2(V) + 8 * D_byte) / B

    where T = total_tokens, V = vocab_size (nominal full vocabulary),
    D_byte = byte_edit_distance_total, B = total_bytes.
    Uses nominal vocabulary size — the full declared vocabulary, not just observed tokens.
    """
    if not total_tokens or not total_bytes or not vocab_size or vocab_size < 2:
        return None
    value = (total_tokens * math.log2(max(int(vocab_size), 2)) + 8 * byte_edit_distance_total) / total_bytes
    return default_format(max(value, 0))


def _split_chars_into_chunks(char_string: str, chunk_count: int) -> list:
    """Split a character string into roughly equal-length chunks."""
    boundaries = [
        round(index * len(char_string) / chunk_count)
        for index in range(chunk_count + 1)
    ]
    return [
        char_string[boundaries[index]:boundaries[index + 1]]
        for index in range(chunk_count)
    ]

def calculate_char_edit_stats(original_text: str, decoded_text: str) -> Tuple[int, int, bool]:
    """Return character-level edit stats, chunking oversized documents to avoid quadratic blowups."""
    chunked = False
    max_length = max(len(original_text), len(decoded_text))

    if original_text == decoded_text:
        return 0, max(max_length, 1), chunked

    levenshtein_distance = get_levenshtein_distance()
    if len(original_text) * len(decoded_text) <= CHAR_FIDELITY_EXACT_MAX_OPS:
        distance = levenshtein_distance(original_text, decoded_text)
        return distance, max(max_length, 1), chunked

    chunked = True
    chunk_count = math.ceil(max_length / CHAR_FIDELITY_CHUNK_CHARS)
    original_chunks = _split_chars_into_chunks(original_text, chunk_count)
    decoded_chunks = _split_chars_into_chunks(decoded_text, chunk_count)

    total_distance = 0
    total_denominator = 0
    for original_chunk, decoded_chunk in zip(original_chunks, decoded_chunks):
        total_distance += levenshtein_distance(original_chunk, decoded_chunk)
        total_denominator += max(len(original_chunk), len(decoded_chunk), 1)

    return total_distance, total_denominator, chunked

def calculate_char_fidelity(char_edit_distance_total: int, char_edit_denominator_total: int) -> float:
    """Calculate character-level reconstruction fidelity from aggregated edit stats."""
    if char_edit_denominator_total <= 0:
        return 0
    fidelity = 1 - (char_edit_distance_total / char_edit_denominator_total)
    return round(max(fidelity, 0), 5)

def calculate_effective_chars_per_token(compression_chars: float, char_fidelity: float) -> float:
    """Calculate effective chars/token after discounting lossy decoding."""
    return default_format(compression_chars * char_fidelity)

def calculate_ebpc(vocab_size: int, effective_chars_per_token: float) -> float:
    """Calculate Effective Bits Per Char. Lower is better."""
    if vocab_size is None or vocab_size <= 1 or effective_chars_per_token <= 0:
        return 0
    return default_format(math.log2(vocab_size) / effective_chars_per_token)


_ZIPF_EMPTY = {
    'freq_cardinality': 0, 'freq_cardinality_ratio': 0.0, 'freq_auc': 0.0, 'freq_slope': 0.0, 'freq_power_law': 0.0,
    'freq_entropy': 0.0, 'freq_renyi_entropy': 0.0,
    'freq_shannon_efficiency': 0.0, 'freq_renyi_efficiency': 0.0,
    'freq_shannon_efficiency_full': 0.0, 'freq_renyi_efficiency_full': 0.0,
    'freq_percentile_freq': 0.0,
    'freq_hapax_count': 0, 'freq_hapax_ratio': 0.0,
    'freq_half_mass_count': 0, 'freq_90pct_mass_count': 0, 'freq_90pct_mass_ratio': 0.0, 'freq_gini': 0.0,
}

def compute_zipf_metrics(token_counter: Counter, vocab_size: int = None) -> dict:
    """
    Compute token distribution metrics from a frequency Counter.
    ref: Lotz et al. (2025) "Beyond Text Compression: Evaluating Tokenizers Across Scales"
         https://arxiv.org/abs/2506.03101
         Zouhar et al. (ACL 2023) "Tokenization and the Noiseless Channel"
         https://aclanthology.org/2023.acl-long.284/

    Metrics computed on the log-log rank-frequency distribution (Lotz et al., 2025):
      freq_cardinality   : number of unique token IDs observed
      freq_cardinality_ratio : unique token IDs observed / tokenizer vocab size
      freq_auc           : area under the log-rank / log-freq curve (Simpson's rule)
      freq_slope         : slope of a linear fit on that curve (approximates Zipf exponent)
      freq_power_law     : mean absolute error from that linear fit (deviation from Zipf)

    Distribution shape metrics (cannot be recomputed from saved scalars):
      freq_entropy            : Shannon entropy in bits of the token probability distribution
      freq_renyi_entropy      : Rényi entropy of order 3 in bits; more sensitive to high-freq
                                tokens than Shannon; alpha=3 best correlates with MT performance
                                (Zouhar et al., ACL 2023 / tokenization-scorer package default)
      freq_shannon_efficiency : Shannon entropy / log2(observed vocab size); comparable across
                                tokenizers with different vocabulary sizes (higher is better)
      freq_renyi_efficiency   : Rényi entropy (alpha=3) / log2(observed vocab size)
      freq_shannon_efficiency_full : Shannon entropy / log2(tokenizer.vocab_size)
      freq_renyi_efficiency_full   : Rényi entropy (alpha=3) / log2(tokenizer.vocab_size)
      freq_percentile_freq    : sum of token probabilities from 3rd to 83rd percentile
                                (descending frequency order); captures mid-frequency mass,
                                second-best predictor of MT performance after Rényi efficiency
      freq_hapax_count   : number of unique token types observed exactly once
      freq_hapax_ratio   : fraction of unique tokens that appear exactly once
      freq_half_mass_count : number of unique token types needed to cover 50% of all
                           token occurrences (smaller = more head-concentrated)
      freq_90pct_mass_ratio : freq_90pct_mass_count / tokenizer vocab size
      freq_gini          : Gini inequality coefficient of the frequency distribution
                           (0=uniform over all types, 1=single dominant token)

    Args:
      token_counter: Counter mapping token IDs to their frequency counts.
      vocab_size: Total tokenizer vocabulary size (tokenizer.vocab_size). When provided,
                  also computes freq_shannon_efficiency_full and freq_renyi_efficiency_full
                  normalized by the full vocab (including unseen tokens).

    The paper restricts the Zipf fit to log(rank) <= 6 (~first 403 ranks) where the
    power-law behaviour is most reliable.
    """
    if not token_counter:
        return _ZIPF_EMPTY.copy()

    cardinality = len(token_counter)
    freq_cardinality_ratio = round(float(cardinality / vocab_size), 5) if vocab_size and vocab_size > 0 else 0.0
    freqs = np.array(sorted(token_counter.values(), reverse=True), dtype=np.float64)
    log_ranks = np.log(np.arange(1, len(freqs) + 1, dtype=np.float64))
    log_freqs = np.log(np.maximum(freqs, 1.0))

    mask = log_ranks <= 6.0  # paper cutoff: ~first 403 unique tokens
    if mask.sum() < 2:
        auc, slope, power_law = 0.0, 0.0, 0.0
    else:
        lr, lf = log_ranks[mask], log_freqs[mask]
        beta1, beta0 = np.polyfit(lr, lf, 1)
        auc = round(float(simpson(lf, x=lr)), 5)
        slope = round(float(beta1), 5)
        power_law = round(float(np.mean(np.abs(beta0 + beta1 * lr - lf))), 5)

    total = freqs.sum()
    probs = freqs / total

    # Shannon entropy
    freq_entropy = round(float(-np.sum(probs * np.log2(np.maximum(probs, 1e-30)))), 5)

    # Rényi entropy of order 3: 1/(1-3) * log2(sum(p_i^3))
    freq_renyi_entropy = round(float(1.0 / (1.0 - 3.0) * np.log2(np.sum(probs ** 3.0))), 5)

    # Shannon efficiency normalized by observed vocab size
    freq_shannon_efficiency = round(float(freq_entropy / np.log2(cardinality)), 5) if cardinality > 1 else 0.0

    # Rényi efficiency (alpha=3) normalized by observed vocab size
    freq_renyi_efficiency = round(float(freq_renyi_entropy / np.log2(cardinality)), 5) if cardinality > 1 else 0.0

    # Full-vocab efficiency variants (normalized by tokenizer.vocab_size)
    freq_shannon_efficiency_full = round(float(freq_entropy / np.log2(vocab_size)), 5) if vocab_size and vocab_size > 1 else 0.0
    freq_renyi_efficiency_full = round(float(freq_renyi_entropy / np.log2(vocab_size)), 5) if vocab_size and vocab_size > 1 else 0.0

    # Percentile frequency: probability mass from 3rd to 83rd percentile (descending order)
    n_probs = len(probs)
    start_i = int(n_probs * 0.03)
    end_i = int(n_probs * 0.83)
    if start_i >= end_i:
        start_i = max(0, end_i - 1)
    freq_percentile_freq = round(float(np.sum(probs[start_i:end_i])), 5)

    # Hapax legomena
    freq_hapax_count = int(np.sum(freqs == 1.0))
    freq_hapax_ratio = round(float(freq_hapax_count / cardinality), 5)

    # Half-mass count: fewest top types that cover >= 50% of all occurrences
    cumulative = np.cumsum(freqs)
    freq_half_mass_count = int(np.searchsorted(cumulative, 0.5 * total) + 1)
    # 90%-mass count: fewest top types that cover >= 90% of all occurrences
    freq_90pct_mass_count = int(np.searchsorted(cumulative, 0.9 * total) + 1)
    freq_90pct_mass_ratio = round(float(freq_90pct_mass_count / vocab_size), 5) if vocab_size and vocab_size > 0 else 0.0

    # Gini coefficient
    sorted_f = np.sort(freqs)
    n = len(sorted_f)
    index = np.arange(1, n + 1, dtype=np.float64)
    freq_gini = round(float((2.0 * np.sum(index * sorted_f) / (n * sorted_f.sum())) - (n + 1.0) / n), 5)

    return {
        'freq_cardinality': int(cardinality),
        'freq_cardinality_ratio': freq_cardinality_ratio,
        'freq_auc': auc,
        'freq_slope': slope,
        'freq_power_law': power_law,
        'freq_entropy': freq_entropy,
        'freq_renyi_entropy': freq_renyi_entropy,
        'freq_shannon_efficiency': freq_shannon_efficiency,
        'freq_renyi_efficiency': freq_renyi_efficiency,
        'freq_shannon_efficiency_full': freq_shannon_efficiency_full,
        'freq_renyi_efficiency_full': freq_renyi_efficiency_full,
        'freq_percentile_freq': freq_percentile_freq,
        'freq_hapax_count': freq_hapax_count,
        'freq_hapax_ratio': freq_hapax_ratio,
        'freq_half_mass_count': freq_half_mass_count,
        'freq_90pct_mass_count': freq_90pct_mass_count,
        'freq_90pct_mass_ratio': freq_90pct_mass_ratio,
        'freq_gini': freq_gini,
    }

# New batched evaluation functions
def load_texts_batched(files_info, batch_size=DEFAULT_BATCH_SIZE):
    """
    Generator that yields batches of texts from multiple lang_domain files.
    
    Args:
        files_info: List of file info dictionaries with 'file_path', 'lang', 'domain', etc.
        batch_size: Maximum number of texts to load in each batch
        
    Yields:
        tuple: (batch_texts, batch_metadata) where batch_metadata contains mapping info
    """
    current_batch_texts = []
    current_batch_metadata = []
    
    for file_info in files_info:
        file_path = file_info['file_path']
        lang = file_info['lang']
        domain = file_info['domain']
        
        file_texts = []
        file_word_counts = []
        file_char_counts = []
        file_byte_counts = []

        # Load all texts from current file
        for entry in yield_jsonl_gz(file_path):
            text = entry['text'].strip()
            if text == '':
                continue
            word_count = entry.get('word_count', len(text.split()))
            char_count = entry.get('char_count', len(text))
            byte_count = entry.get('byte_count', len(text.encode('utf-8')))

            file_texts.append(text)
            file_word_counts.append(word_count)
            file_char_counts.append(char_count)
            file_byte_counts.append(byte_count)
        
        # Add texts from this file to current batch
        start_idx = len(current_batch_texts)
        current_batch_texts.extend(file_texts)
        
        # Store metadata for this file's contribution to the batch
        current_batch_metadata.append({
            'lang': lang,
            'domain': domain,
            'file_info': file_info,
            'start_idx': start_idx,
            'end_idx': len(current_batch_texts),
            'texts': file_texts,
            'word_counts': file_word_counts,
            'char_counts': file_char_counts,
            'byte_counts': file_byte_counts,
            'total_docs': len(file_texts),
            'total_words': sum(file_word_counts),
            'total_chars': sum(file_char_counts),
            'total_bytes': sum(file_byte_counts)
        })
        
        # Yield batch if it's large enough
        if len(current_batch_texts) >= batch_size:
            yield current_batch_texts, current_batch_metadata
            current_batch_texts = []
            current_batch_metadata = []
    
    # Yield remaining texts if any
    if current_batch_texts:
        yield current_batch_texts, current_batch_metadata

def process_batch_results(batch_texts, batch_metadata, batch_encoding, unencoded_texts, tokenizer):
    """
    Process the results of a batch tokenization and map back to individual lang_domains.
    
    Args:
        batch_texts: List of original texts in the batch
        batch_metadata: List of metadata for each file's contribution to batch
        batch_encoding: Tokenizer encoding result for the batch
        unencoded_texts: Decoded texts from tokenizer
        tokenizer: The tokenizer instance
        
    Returns:
        Tuple of:
          - List of result dictionaries, one per lang_domain
          - Dict mapping (lang, domain) -> Counter of token-id frequencies for that domain
    """
    unk_token_id = tokenizer.unk_token_id
    results = []
    domain_counters: Dict[Tuple[str, str], Counter] = {}
    
    for file_meta in batch_metadata:
        start_idx = file_meta['start_idx']
        end_idx = file_meta['end_idx']
        
        # Extract this file's portion from batch results
        file_token_lists = batch_encoding.input_ids[start_idx:end_idx]
        file_unencoded_texts = unencoded_texts[start_idx:end_idx]
        file_original_texts = batch_texts[start_idx:end_idx]
        
        # Calculate metrics for this file
        total_tokens = sum(len(tokens) for tokens in file_token_lists)
        
        # Build token-frequency counter and compute Zipf metrics
        token_counter: Counter = Counter()
        for token_list in file_token_lists:
            token_counter.update(token_list)
        zipf_metrics = compute_zipf_metrics(token_counter, vocab_size=tokenizer.vocab_size)
        domain_counters[(file_meta['lang'], file_meta['domain'])] = token_counter

        # Count UNK tokens efficiently
        unk_tokens_count = 0
        if unk_token_id is not None:
            for tokens in file_token_lists:
                unk_tokens_count += np.sum(np.array(tokens) == unk_token_id)
        
        # Count reversible documents
        is_reversible_texts = [
            orig == decoded 
            for orig, decoded in zip(file_original_texts, file_unencoded_texts)
        ]
        reversible_docs_count = sum(is_reversible_texts)

        byte_edit_distance_total = 0
        byte_edit_denominator_total = 0
        decoded_bytes_total = 0
        char_edit_distance_total = 0
        char_edit_denominator_total = 0
        decoded_chars_total = 0
        for orig, decoded in zip(file_original_texts, file_unencoded_texts):
            b_dist, b_denom, _ = calculate_byte_edit_stats(orig, decoded)
            byte_edit_distance_total += b_dist
            byte_edit_denominator_total += b_denom
            decoded_bytes_total += len(decoded.encode('utf-8'))
            c_dist, c_denom, _ = calculate_char_edit_stats(orig, decoded)
            char_edit_distance_total += c_dist
            char_edit_denominator_total += c_denom
            decoded_chars_total += len(decoded)
        
        # Calculate metrics
        fertility = calculate_fertility(total_tokens, file_meta['total_words'])
        compression_chars = calculate_compression(file_meta['total_chars'], total_tokens)
        compression_bytes = calculate_compression(file_meta['total_bytes'], total_tokens)
        reversible_ratio = calculate_reversible_ratio(reversible_docs_count, file_meta['total_docs'])
        unk_ratio = calculate_unk_ratio(unk_tokens_count, total_tokens)
        byte_fidelity = calculate_byte_fidelity(byte_edit_distance_total, byte_edit_denominator_total)
        effective_bytes_per_token = calculate_effective_bytes_per_token(compression_bytes, byte_fidelity)
        ebpb = calculate_ebpb(total_tokens, file_meta['total_bytes'], byte_edit_distance_total, zipf_metrics.get('freq_cardinality'))
        ebpb_full = calculate_ebpb_full(total_tokens, file_meta['total_bytes'], byte_edit_distance_total, tokenizer.vocab_size)
        char_fidelity = calculate_char_fidelity(char_edit_distance_total, char_edit_denominator_total)
        effective_chars_per_token = calculate_effective_chars_per_token(compression_chars, char_fidelity)
        ebpc = calculate_ebpc(tokenizer.vocab_size, effective_chars_per_token)
        
        # Calculate file hash
        benchmark_file_hash = file_meta['file_info']['benchmark_file_hash']
        evaluation_date = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        result = {
            "lang": file_meta['lang'],
            "domain": file_meta['domain'],
            "evaluation_type": "standard",
            "fertility": fertility,
            "compression_chars": compression_chars,
            "compression_bytes": compression_bytes,
            "reversible_ratio": reversible_ratio,
            "unk_ratio": unk_ratio,
            "byte_fidelity": byte_fidelity,
            "effective_bytes_per_token": effective_bytes_per_token,
            "ebpb": ebpb,
            "ebpb_full": ebpb_full,
            "char_fidelity": char_fidelity,
            "effective_chars_per_token": effective_chars_per_token,
            "ebpc": ebpc,
            "total_tokens": total_tokens,
            "total_words": file_meta['total_words'],
            "total_chars": file_meta['total_chars'],
            "total_bytes": file_meta['total_bytes'],
            'reversible_docs_count': reversible_docs_count,
            'unk_tokens_count': unk_tokens_count,
            'byte_edit_distance_total': byte_edit_distance_total,
            'byte_edit_denominator_total': byte_edit_denominator_total,
            'decoded_bytes_total': decoded_bytes_total,
            'char_edit_distance_total': char_edit_distance_total,
            'char_edit_denominator_total': char_edit_denominator_total,
            'decoded_chars_total': decoded_chars_total,
            'total_docs': file_meta['total_docs'],
            "benchmark_file_hash": benchmark_file_hash,
            "evaluation_date": evaluation_date,
            "reference": False,
            **zipf_metrics,
        }
        
        results.append(result)
    
    return results, domain_counters

def evaluate_lang_domains_batched(tokenizer, files_to_evaluate, batch_size=DEFAULT_BATCH_SIZE, _target_mode=False, verbose=False):
    """
    Evaluate multiple lang_domain files using batched processing for efficiency.
    
    Args:
        tokenizer: The tokenizer to evaluate
        files_to_evaluate: List of file info dictionaries
        batch_size: Maximum number of texts to process in each batch
        verbose: Whether to print progress information
        
    Returns:
        Tuple of:
          - List of result dictionaries, one per lang_domain
          - Dict mapping lang -> merged Counter of token-id frequencies across all evaluated domains
    """
    if verbose:
        print(f"Starting batched evaluation with batch_size={batch_size}")
        print(f"Processing {len(files_to_evaluate)} files")
    
    all_results = []
    lang_counters: Dict[str, Counter] = {}
    batch_count = 0
    
    # Set tokenizer to avoid truncation warnings
    tokenizer.model_max_length = int(1e9)
    
    for batch_texts, batch_metadata in load_texts_batched(files_to_evaluate, batch_size):
        batch_count += 1
        start_time = time.time()
        
        if verbose:
            files_in_batch = [meta['lang'] + '_' + meta['domain'] for meta in batch_metadata]
            print(f"  Batch {batch_count}: {len(batch_texts)} texts from {len(batch_metadata)} files: {', '.join(files_in_batch)}")
        
        # Tokenize entire batch
        encode_start = time.time()
        if not _target_mode:
            encoding = tokenizer(
                batch_texts, 
                add_special_tokens=False, 
                padding=False, 
                truncation=False,
                return_token_type_ids=False,
                return_attention_mask=False 
                #return_tensors='np'
            )
        else:
            encoding = tokenizer(
                text_target=batch_texts, 
                add_special_tokens=False, 
                padding=False, 
                truncation=False, 
                return_token_type_ids=False,
                return_attention_mask=False,
                #return_tensors='np'
            )
        encode_time = time.time() - encode_start
        
        # Decode entire batch
        decode_start = time.time()
        unencoded_texts = tokenizer.decode(encoding.input_ids, skip_special_tokens=False)
        decode_time = time.time() - decode_start
        
        # Process results for each file in the batch
        batch_results, domain_counters = process_batch_results(
            batch_texts, batch_metadata, encoding, unencoded_texts, tokenizer
        )
        
        # Accumulate per-language counters
        for (lang, _domain), counter in domain_counters.items():
            if lang not in lang_counters:
                lang_counters[lang] = Counter()
            lang_counters[lang] += counter

        all_results.extend(batch_results)
        
        batch_time = time.time() - start_time
        if verbose:
            print(f"    Batch {batch_count} completed in {batch_time:.2f}s (encode: {encode_time:.2f}s, decode: {decode_time:.2f}s)")
    
    if verbose:
        print(f"Batched evaluation completed: {len(all_results)} results from {batch_count} batches")
    
    return all_results, lang_counters

def collect_frequencies_batched(tokenizer, files_info, batch_size=DEFAULT_BATCH_SIZE, _target_mode=False):
    """
    Lightweight tokenize-only pass: returns per-language token-frequency Counters
    without performing decoding or computing any other metrics.

    Used to gather frequency data for domains that were already cached (not freshly
    evaluated) so that per-language Zipf metrics can be computed from all domains.

    Args:
        tokenizer: The tokenizer to use
        files_info: List of file info dicts (same format as files_to_evaluate)
        batch_size: Texts per tokenization batch
        _target_mode: Whether to tokenize as target (seq2seq tokenizers)

    Returns:
        Dict mapping lang -> merged Counter of token-id frequencies
    """
    lang_counters: Dict[str, Counter] = {}
    tokenizer.model_max_length = int(1e9)
    for batch_texts, batch_metadata in load_texts_batched(files_info, batch_size):
        if not _target_mode:
            encoding = tokenizer(
                batch_texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            )
        else:
            encoding = tokenizer(
                text_target=batch_texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_token_type_ids=False,
                return_attention_mask=False,
            )
        for file_meta in batch_metadata:
            start_idx = file_meta['start_idx']
            end_idx = file_meta['end_idx']
            lang = file_meta['lang']
            counter: Counter = Counter()
            for token_list in encoding.input_ids[start_idx:end_idx]:
                counter.update(token_list)
            if lang not in lang_counters:
                lang_counters[lang] = Counter()
            lang_counters[lang] += counter
    return lang_counters


# Evaluate a tokenizer on a single lang_domain file
def evaluate_lang_domain(tokenizer, file_path, meta_info, _target_mode=False, verbose=False):
    """Evaluate a tokenizer on a single lang_domain file."""
    if verbose:
        print(f"  Processing domain: {meta_info['domain']}")
    general_time = time.time()

    # Calculate file hash
    benchmark_file_hash = calculate_file_hash(file_path)
    time_file_hash = time.time() - general_time
    if verbose:
        print(f"    File hash: {benchmark_file_hash}")
    
    # Collect all texts and word counts
    all_texts = []
    total_words = 0
    total_chars = 0
    total_bytes = 0
    evaluation_date = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # First pass: collect all texts and count words
    start_time = time.time()
    for entry in yield_jsonl_gz(file_path):
        text = entry['text'].strip()
        if text == '':
            continue
        word_count = entry.get('word_count', len(text.split()))
        char_count = entry.get('char_count', len(text))
        byte_count = entry.get('byte_count', len(text.encode('utf-8')))
        all_texts.append(text)
        total_words += word_count
        total_chars += char_count
        total_bytes += byte_count

    total_docs = len(all_texts)
    time_load = time.time() - start_time
    
    start_time = time.time()
    tokenizer.model_max_length = int(1e9)  # Set a very high max length to avoid truncation warninigs
    if not _target_mode:
        encoding = tokenizer(all_texts, add_special_tokens=False, padding=False, truncation=False)
    else:
        encoding = tokenizer(text_target=all_texts, add_special_tokens=False, padding=False, truncation=False)
    unk_token_id = tokenizer.unk_token_id
    time_encode = time.time() - start_time
    
    start_time = time.time()
    total_tokens = sum(len(tokens) for tokens in encoding.input_ids)
    time_count_tokens = time.time() - start_time

    start_time = time.time()
    #slow
    #unk_tokens_count = sum([sum(1 for token in tokens if token == tokenizer.unk_token_id) for tokens in encoding.input_ids])
    
    #faster
    #unk_tokens_count = 0
    #for tokens in encoding.input_ids:
    #   for token in tokens:
    #       if token == unk_token_id:
    #           unk_tokens_count += 1

    #numpy count
    unk_tokens_count = np.sum(np.concatenate(encoding.input_ids) == unk_token_id)
    time_count_unk = time.time() - start_time

    start_time = time.time()
    unencoded_texts = tokenizer.decode(encoding.input_ids)
    time_decode = time.time() - start_time

    start_time = time.time()
    is_reversible_texts = [text == unencoded_text for text, unencoded_text in zip(all_texts, unencoded_texts)]
    reversible_docs_count = sum(is_reversible_texts)
    time_count_reversible = time.time() - start_time

    start_time = time.time()
    byte_edit_distance_total = 0
    byte_edit_denominator_total = 0
    decoded_bytes_total = 0
    char_edit_distance_total = 0
    char_edit_denominator_total = 0
    decoded_chars_total = 0
    chunked_count = 0
    for text, unencoded_text in zip(all_texts, unencoded_texts):
        b_dist, b_denom, chunked = calculate_byte_edit_stats(text, unencoded_text)
        byte_edit_distance_total += b_dist
        byte_edit_denominator_total += b_denom
        decoded_bytes_total += len(unencoded_text.encode('utf-8'))
        c_dist, c_denom, _ = calculate_char_edit_stats(text, unencoded_text)
        char_edit_distance_total += c_dist
        char_edit_denominator_total += c_denom
        decoded_chars_total += len(unencoded_text)
        if chunked:
            chunked_count += 1
    chunked_ratio = chunked_count / total_docs if total_docs > 0 else 0
    time_count_byte_fidelity = time.time() - start_time
    
    fertility = calculate_fertility(total_tokens, total_words)
    compression_chars = calculate_compression(total_chars, total_tokens)
    compression_bytes = calculate_compression(total_bytes, total_tokens)
    reversible_ratio = calculate_reversible_ratio(reversible_docs_count, total_docs)
    unk_ratio = calculate_unk_ratio(unk_tokens_count, total_tokens)
    all_ids = np.concatenate(encoding.input_ids)
    token_counter = Counter(all_ids.tolist())
    zipf_metrics = compute_zipf_metrics(token_counter, vocab_size=tokenizer.vocab_size)

    byte_fidelity = calculate_byte_fidelity(byte_edit_distance_total, byte_edit_denominator_total)
    effective_bytes_per_token = calculate_effective_bytes_per_token(compression_bytes, byte_fidelity)
    ebpb = calculate_ebpb(total_tokens, total_bytes, byte_edit_distance_total, zipf_metrics.get('freq_cardinality'))
    ebpb_full = calculate_ebpb_full(total_tokens, total_bytes, byte_edit_distance_total, tokenizer.vocab_size)
    char_fidelity = calculate_char_fidelity(char_edit_distance_total, char_edit_denominator_total)
    effective_chars_per_token = calculate_effective_chars_per_token(compression_chars, char_fidelity)
    ebpc = calculate_ebpc(tokenizer.vocab_size, effective_chars_per_token)

    time_lang = time.time() - general_time

    if verbose:
        print(f"    Fertility: {fertility}, Tokens: {total_tokens}, Words: {total_words}")
        print(f"    Compression (Chars): {compression_chars}, Compression (Bytes): {compression_bytes}, Chars: {total_chars}, Bytes: {total_bytes}")
        print(f"    Reversible: {reversible_ratio}, Count: {reversible_docs_count}, Total Docs: {total_docs}")
        print(f"    Unk Ratio: {unk_ratio}, Count: {unk_tokens_count}")
        print(f"    Byte Fidelity: {byte_fidelity}, Effective Bytes/Token: {effective_bytes_per_token}, EBPB: {ebpb}")
        print(f"    Char Fidelity: {char_fidelity}, Effective Chars/Token: {effective_chars_per_token}, EBPC: {ebpc}")
        print(f"    Time steps: Total: {time_lang:.2f}, File Hash: {time_file_hash:.5f}, Load: {time_load:.5f}, Encode: {time_encode:.5f}, Count Tokens: {time_count_tokens:.5f}, Count Unk: {time_count_unk:.5f}, Decode: {time_decode:.5f}, Count Reversible: {time_count_reversible:.5f}, Byte Fidelity: {time_count_byte_fidelity:.5f}, Chunked: {chunked_ratio:.3f}")

    return {
        "lang": meta_info['lang'],
        "domain": meta_info['domain'],
        "evaluation_type": "standard",
        "fertility": fertility,
        "compression_chars": compression_chars,
        "compression_bytes": compression_bytes,
        "reversible_ratio": reversible_ratio,
        "unk_ratio": unk_ratio,
        "byte_fidelity": byte_fidelity,
        "effective_bytes_per_token": effective_bytes_per_token,
        "ebpb": ebpb,
        "ebpb_full": ebpb_full,
        "char_fidelity": char_fidelity,
        "effective_chars_per_token": effective_chars_per_token,
        "ebpc": ebpc,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        'reversible_docs_count': reversible_docs_count,
        'unk_tokens_count': unk_tokens_count,
        'byte_edit_distance_total': byte_edit_distance_total,
        'byte_edit_denominator_total': byte_edit_denominator_total,
        'decoded_bytes_total': decoded_bytes_total,
        'char_edit_distance_total': char_edit_distance_total,
        'char_edit_denominator_total': char_edit_denominator_total,
        'decoded_chars_total': decoded_chars_total,
        'total_docs': total_docs,
        "benchmark_file_hash": benchmark_file_hash,
        "evaluation_date": evaluation_date,
        #"evaluation_time": time_lang,
        "reference": False,
        **zipf_metrics,
    }, token_counter

# Calculate overall results for a single model's data
def calculate_overall_for_model(df, meta=None):
    """
    Calculate various overall metrics for a model's results.
    
    This function now delegates to the optimized vectorized implementation for better performance.
    
    Args:
        df: DataFrame with model results
        tokenizer_meta: Optional tokenizer metadata
        meta: Optional dataset metadata from dataset_meta.yaml to calculate multilingual overalls
        
    Returns:
        Dictionary with lang_overall entries for each language and 
        potentially a multilingual_overall entry if all languages were evaluated
    """
    # Import dynamically to avoid circular import
    from results_aggregator import calculate_overall_for_model_vectorized
    
    # Delegate to the optimized vectorized implementation
    return calculate_overall_for_model_vectorized(df, meta)

# ---------------------------------------------
# Results I/O helpers
# ---------------------------------------------

# Helper: build a safe filename stem for model names (replace '/')
def _sanitize_text(text: str) -> str:
    """Return a filesystem-safe version of the model name."""
    return text.replace('/', '_')

def _get_results_eval_filename(model_name: str, revision: str, subfolder: str = None, _target_mode: bool = False) -> str:
    """Get the evaluation filename for a model."""
    model_key = generate_model_key(model_name, revision, subfolder, _target_mode)
    return f"results_{model_key}.jsonl"

def _get_meta_eval_filename(model_name: str, revision: str, subfolder: str = None, _target_mode: bool = False) -> str:
    """Get the evaluation filename for a model."""
    model_key = generate_model_key(model_name, revision, subfolder, _target_mode)
    return f"meta_{model_key}.json"

def _load_model_file(filepath):
    df = pd.read_json(filepath, lines=True)
    # Create model_key using the centralized function
    def create_model_key(row):
        revision = row['revision'] if pd.notna(row['revision']) else 'main'
        subfolder = row.get('subfolder') if pd.notna(row.get('subfolder')) else None
        _target_mode = row.get('_target_mode') if pd.notna(row.get('_target_mode')) else False
        return generate_model_key(row['model'], revision, subfolder, _target_mode)
    
    df['model_key'] = df.apply(create_model_key, axis=1)
    return df

# Load existing results for a specific model and vocab hash
def load_model_results(model_name, revision="main", subfolder=None, _target_mode=False, results_dir=None):
    """Load existing results for a specific model and vocab hash if available."""
    if revision is None:
        raise ValueError("revision must be provided under the new filename convention.")

    if results_dir is None:
        results_dir = Path(DATA_DIR) / DEFAULT_RESULTS_PATH / DEFAULT_MODEL_EVALUATIONS_PATH
        results_dir.mkdir(exist_ok=True, parents=True)

    results_path = results_dir / DEFAULT_MODEL_EVALUATIONS_PATH / _get_results_eval_filename(model_name, revision, subfolder, _target_mode)

    if results_path.exists():
        return _load_model_file(results_path)

    return pd.DataFrame()

    """
    try:
        if results_path.exists():
            return pd.read_json(results_path, lines=True)

        # Try to fetch from hub if not present locally
        file_path = hf_hub_download(
            repo_id=HF_REPO_RESULTS,
            filename=results_path.name,
            local_dir=results_dir,
            repo_type="dataset",
        )
        return pd.read_json(file_path, lines=True)
    except Exception as e:
        # If the file doesn't exist on the hub or any other error occurs, return empty DataFrame
        if isinstance(e, FileNotFoundError):
            return pd.DataFrame()
        print(f"Error loading existing results: {e}")
        return pd.DataFrame()
    """

# Save model results to disk and upload to HF Hub
def save_model_results(df, model_name, revision="main", subfolder=None, _target_mode=False, results_dir=None, upload_to_hub: bool = False, output_path: str | Path | None = None, verbose=False):
    """Save model results to disk and optionally upload to HF Hub."""
    if df.empty:
        return False
    
    if results_dir is None:
        results_dir = Path(DATA_DIR) / DEFAULT_RESULTS_PATH
    
    evaluations_dir = results_dir / DEFAULT_MODEL_EVALUATIONS_PATH
    evaluations_dir.mkdir(exist_ok=True, parents=True)

    results_path = evaluations_dir / _get_results_eval_filename(model_name, revision, subfolder, _target_mode)
    df.to_json(results_path, lines=True, orient="records", force_ascii=False)
    
    if verbose:
        print(f"\nResults saved to {results_path}")
    
    # Copy to output_path if specified
    if output_path:
        try:
            output_target_path = Path(output_path)
            output_target_path.mkdir(parents=True, exist_ok=True)  # Ensure target directory exists
            output_target_filepath = output_target_path / results_path.name
            shutil.copy2(results_path, output_target_filepath) # copy2 preserves metadata
            if verbose:
                print(f"Results also copied to {output_target_filepath}")
        except Exception as e:
            print(f"Error copying results to {output_path}: {e}")

    # Optionally upload to HF hub
    if upload_to_hub:
        api.upload_file(
            path_or_fileobj=results_path,
            path_in_repo=str( Path(DEFAULT_MODEL_EVALUATIONS_PATH) / results_path.name),
            repo_id=HF_REPO_RESULTS,
            repo_type="dataset",
        )
        
        if verbose:
            print(f"Uploaded {results_path.name} to {HF_REPO_RESULTS} on the Hub.")
    
    return True

def get_tokenizer_config(tokenizer, vocab_hash=None):

    _safe_get_attr = lambda obj, attr: getattr(obj, attr) if hasattr(obj, attr) else None

    def _get_from_obj_or_init_kwargs(obj, attr):
        obj_res = _safe_get_attr(obj, attr)
        if obj_res is None and _safe_get_attr(obj, 'init_kwargs') is not None:
            if attr in obj.init_kwargs:
                obj_res = obj.init_kwargs[attr]
        return obj_res

    def _deep_sort_json(json_data):
        if isinstance(json_data, dict):
            return {k: _deep_sort_json(v) for k, v in sorted(json_data.items())}
        elif isinstance(json_data, list):
            return [_deep_sort_json(item) for item in json_data]
        else:
            return json_data
        
    def _sp_model_hash(sp_model):
        if sp_model is None:
            return None
        try:
            return hashlib.md5(sp_model.serialized_model_proto()).hexdigest()
        except Exception as e:
            return None
        
    def _hash_precompiled_charsmap(data):
        if isinstance(data, dict):
            if 'precompiled_charsmap' in data:
                data['precompiled_charsmap_hash'] = hashlib.md5(data['precompiled_charsmap'].encode('utf-8')).hexdigest()
                del data['precompiled_charsmap']
            for value in data.values():
                _hash_precompiled_charsmap(value)
        elif isinstance(data, list):
            for item in data:
                _hash_precompiled_charsmap(item)
        return
    
    if vocab_hash is None:
        vocab_hash, _ = calculate_vocab_hash(tokenizer)

    tokenizer_config = {
        'tokenizer_class': type(tokenizer).__name__,
        'tokenizer.is_fast': _safe_get_attr(tokenizer, 'is_fast'),
        'tokenizer.backend_tokenizer': None,
        'tokenizer.backend_tokenizer.model': None,

        'tokenizer.backend_tokenizer.normalizer': None,
        'tokenizer.backend_tokenizer.pre_tokenizer': None,
        'tokenizer.backend_tokenizer.post_processor': None,
        'tokenizer.backend_tokenizer.decoder': None,

        'tokenizer.backend_tokenizer.model.continuing_subword_prefix': None,
        'tokenizer.backend_tokenizer.model.end_of_word_suffix': None,
        'tokenizer.backend_tokenizer.model.fuse_unk': None,
        'tokenizer.backend_tokenizer.model.byte_fallback': None,
        'tokenizer.backend_tokenizer.model.ignore_merges': None,
        'tokenizer.backend_tokenizer.model.max_input_chars_per_word': None,

        # The multiple configurations that differents tokenizers can have and may affect the encoding/decoding result
        'tokenizer.legacy': _get_from_obj_or_init_kwargs(tokenizer, 'legacy'),
        'tokenizer.legacy_behaviour': _get_from_obj_or_init_kwargs(tokenizer, 'legacy_behaviour'),
        'tokenizer.clean_up_tokenization_spaces': _get_from_obj_or_init_kwargs(tokenizer, 'clean_up_tokenization_spaces'),
        'tokenizer.add_prefix_space': _get_from_obj_or_init_kwargs(tokenizer, 'add_prefix_space'),
        'tokenizer.do_lower_case': _get_from_obj_or_init_kwargs(tokenizer, 'do_lower_case'),
        'tokenizer.do_lowercase': _get_from_obj_or_init_kwargs(tokenizer, 'do_lowercase'),
        'tokenizer.lower_case': _get_from_obj_or_init_kwargs(tokenizer, 'lower_case'),
        'tokenizer.remove_space': _get_from_obj_or_init_kwargs(tokenizer, 'remove_space'),
        'tokenizer.strip_accents': _get_from_obj_or_init_kwargs(tokenizer, 'strip_accents'),
        'tokenizer.keep_accents': _get_from_obj_or_init_kwargs(tokenizer, 'keep_accents'),
        'tokenizer.trim_offsets': _get_from_obj_or_init_kwargs(tokenizer, 'trim_offsets'),
        'tokenizer.do_lowercase_and_remove_accent': _get_from_obj_or_init_kwargs(tokenizer, 'do_lowercase_and_remove_accent'),
        'tokenizer.errors': _get_from_obj_or_init_kwargs(tokenizer, 'errors'),
        'tokenizer.clean_text': _get_from_obj_or_init_kwargs(tokenizer, 'clean_text'),
        'tokenizer.do_clean_text': _get_from_obj_or_init_kwargs(tokenizer, 'do_clean_text'),
        'tokenizer.normalization': _get_from_obj_or_init_kwargs(tokenizer, 'normalization'),
        'tokenizer.do_basic_tokenize': _get_from_obj_or_init_kwargs(tokenizer, 'do_basic_tokenize'),
        'tokenizer.tokenize_chinese_chars': _get_from_obj_or_init_kwargs(tokenizer, 'tokenize_chinese_chars'),

        'tokenizer.sp_model_kwargs': _deep_sort_json(_safe_get_attr(tokenizer, 'sp_model_kwargs')),
        'tokenizer._sp_model_hash': _sp_model_hash(_safe_get_attr(tokenizer, 'sp_model')),
        'tokenizer.current_spm_hash': _sp_model_hash(_safe_get_attr(tokenizer, 'current_spm')),

        'tokenizer.vocab_hash': vocab_hash,
    }

    if _safe_get_attr(tokenizer, 'backend_tokenizer') is not None:
        tokenizer_config['tokenizer.backend_tokenizer'] = type(tokenizer.backend_tokenizer).__name__

        tokenizer_json = None
        try:
            tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str(pretty=True))
            for key in ['normalizer', 'pre_tokenizer', 'post_processor', 'decoder']:
                if key in tokenizer_json:
                    tokenizer_config[f'tokenizer.backend_tokenizer.{key}'] = _deep_sort_json(tokenizer_json[key])
        except Exception as e:
            print(f'Error loading tokenizer.backend_tokenizer: {e}')
            pass

        if _safe_get_attr(tokenizer.backend_tokenizer, 'model') is not None:
            tokenizer_config['tokenizer.backend_tokenizer.model'] = type(tokenizer.backend_tokenizer.model).__name__

            if tokenizer_json is not None:
                try:
                    model_json = tokenizer_json.get('model', {})
                    for key in ['continuing_subword_prefix', 'end_of_word_suffix', 'fuse_unk', 'byte_fallback', 'ignore_merges', 'max_input_chars_per_word']:
                        if key in model_json:
                            tokenizer_config[f'tokenizer.backend_tokenizer.model.{key}'] = model_json[key]
                except Exception as e:
                    print(f'Error loading tokenizer.backend_tokenizer.model: {e}')
                    pass

    if tokenizer_config['tokenizer.backend_tokenizer.normalizer'] is not None:
        _hash_precompiled_charsmap(tokenizer_config['tokenizer.backend_tokenizer.normalizer'])

    tokenizer_config_hash = hashlib.md5(json.dumps(tokenizer_config, sort_keys=True).encode('utf-8')).hexdigest()

    return tokenizer_config, tokenizer_config_hash

def build_tokenizer_meta(tokenizer, tokenizer_name, vocab_hash, tokenizer_config_hash, tokenizer_config, near_duplicates, filtered_vocab, revision="main", subfolder=None, _target_mode=False):
    """Build tokenizer metadata."""

    if near_duplicates:
        copied_near_duplicates = deepcopy(near_duplicates)
        del copied_near_duplicates['groups_by_step']
    else:
        copied_near_duplicates = None

    algorithm, implementation = get_tokenizer_type(tokenizer)

    data = {
        "model": tokenizer_name,
        "model_name_or_path": tokenizer.name_or_path,
        "revision": revision,
        "subfolder": subfolder,
        "_target_mode": _target_mode,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens_count": tokenizer.vocab_size - len(filtered_vocab),
        "vocab_hash": vocab_hash,
        "tokenizer_config_hash": tokenizer_config_hash,
        "tokenizer_config": tokenizer_config,
        #"init_kwargs": tokenizer.init_kwargs,
        "near_duplicates": copied_near_duplicates,
        'slow_tokenizer_class': tokenizer.slow_tokenizer_class.__name__ if tokenizer.slow_tokenizer_class else None,
        'tokenizer_class': tokenizer.__class__.__name__,
        "tokenizer_algorithm": algorithm,
        "tokenizer_implementation": implementation,
        "evaluation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return data

# Note: save_tokenizer_metadata function removed - it was unused and had bugs
# If tokenizer metadata saving is needed, implement it properly when required

# Build a cache of tokenizer hashes from all existing results files
def build_tokenizer_hash_cache(results_dir=None):
    """
    Build a cache of tokenizer hashes mapping to result files.
    Only stores file paths rather than full result data for memory efficiency.
    
    Returns:
        dict: Map of tokenizer hash → list of result file paths
    """
    if results_dir is None:
        results_dir = Path(DATA_DIR) / DEFAULT_RESULTS_PATH
    
    evaluations_dir = results_dir / DEFAULT_MODEL_EVALUATIONS_PATH

    tokenizer_hash_cache = {}
    result_files = list(Path(evaluations_dir).glob("results_*.jsonl"))
    tokenizer_hash_field = get_tokenizer_hash_field()
    
    for result_file in tqdm(result_files, desc="Building tokenizer hash cache"):
        try:
            # Only read the first few rows to extract the tokenizer hash
            # This is more efficient than loading the entire file
            with open(result_file, 'r') as f:
                first_row = json.loads(f.readline().strip())
                if tokenizer_hash_field in first_row:
                    tokenizer_hash = first_row[tokenizer_hash_field]
                    
                    if tokenizer_hash not in tokenizer_hash_cache:
                        tokenizer_hash_cache[tokenizer_hash] = []
                    
                    # Just store the file path for this hash
                    if result_file not in tokenizer_hash_cache[tokenizer_hash]:
                        tokenizer_hash_cache[tokenizer_hash].append(result_file)
        except Exception as e:
            print(f"Error processing results file {result_file}: {e}")
    
    return tokenizer_hash_cache

# Get results from the tokenizer hash cache
def load_results_by_tokenizer_hash(tokenizer_hash, model_name, revision="main", subfolder=None, _target_mode=False, results_dir=None, tokenizer_hash_cache=None):
    """Get results from the tokenizer hash cache using file paths."""
    # Build or use existing cache
    if tokenizer_hash_cache is None:
        tokenizer_hash_cache = build_tokenizer_hash_cache(results_dir)
    
    if tokenizer_hash in tokenizer_hash_cache:
        # Collect results from all files with this tokenizer hash
        all_results = []
        
        for result_file in tokenizer_hash_cache[tokenizer_hash]:
            try:
                # Load the data from the file
                df = _load_model_file(result_file)
                
                # Update the model name to the current model
                df['model'] = model_name
                df['revision'] = revision
                df['subfolder'] = subfolder
                df['_target_mode'] = _target_mode
                # Create model_key using centralized function
                df['model_key'] = generate_model_key(model_name, revision, subfolder, _target_mode)

                all_results.append(df)
            except Exception as e:
                print(f"Error loading results from {result_file}: {e}")
        
        if all_results:
            # Combine all results into a single DataFrame
            return pd.concat(all_results, ignore_index=True)
    
    return pd.DataFrame()

# Get model results, either from cache or from file
def get_author_info(api, model_name, cache=None):
    """
    Fetch author (org or user) metadata for a given model name from the HF Hub.

    Tries organization first; falls back to user on failure.

    Args:
        api: HfApi instance
        model_name: full model id (e.g. 'deepseek-ai/DeepSeek-V3')
        cache: optional dict mapping author_name -> author_info dict

    Returns:
        dict with keys: author_type, author_name, author_fullname,
                        author_followers, author_models, author_is_verified
    """
    parts = model_name.split('/')
    author_name = parts[0] if len(parts) > 1 else model_name

    if cache is not None and author_name in cache:
        return cache[author_name]

    default = {
        "author_type": None,
        "author_name": author_name,
        "author_fullname": None,
        "author_followers": 0,
        "author_models": 0,
        "author_is_verified": False,
    }

    result = default.copy()
    try:
        org = api.get_organization_overview(author_name)
        result = {
            "author_type": "organization",
            "author_name": author_name,
            "author_fullname": getattr(org, 'fullname', None),
            "author_followers": getattr(org, 'num_followers', 0),
            "author_models": getattr(org, 'num_models', 0),
            "author_is_verified": getattr(org, 'is_verified', False),
        }
    except Exception as e:
        # Re-raise rate-limit errors so callers can retry
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
            raise
        try:
            user = api.get_user_overview(author_name)
            result = {
                "author_type": "user",
                "author_name": author_name,
                "author_fullname": getattr(user, 'fullname', None),
                "author_followers": getattr(user, 'num_followers', 0),
                "author_models": getattr(user, 'num_models', 0),
                "author_is_verified": False,
            }
        except Exception as e2:
            # Re-raise rate-limit errors so callers can retry
            if hasattr(e2, 'response') and e2.response is not None and e2.response.status_code == 429:
                raise e2

    if cache is not None:
        cache[author_name] = result

    return result


def get_model_results(tokenizer_hash, model_name, revision="main", subfolder=None, _target_mode=False, force_rerun=False, verbose=False, results_dir=None, tokenizer_hash_cache=None):
    """Get results for a model, either from cache."""
    existing_results = pd.DataFrame()
    tokenizer_hash_field = get_tokenizer_hash_field()
    
    if not force_rerun:
        # Try to find results with matching tokenizer hash in the cache first
        existing_results = load_results_by_tokenizer_hash(tokenizer_hash, model_name, revision, subfolder, _target_mode, results_dir, tokenizer_hash_cache)
        
        # Deduplicate results loaded from cache based on key identifiers
        if not existing_results.empty:
            if verbose:
                initial_count = len(existing_results)
            # Define columns that uniquely identify an evaluation result
            id_columns = ['lang', 'domain']
            # Ensure all required columns exist before attempting deduplication
            if all(col in existing_results.columns for col in id_columns):
                existing_results = existing_results.drop_duplicates(subset=id_columns, keep='last')
                if verbose:
                    final_count = len(existing_results)
                    if initial_count != final_count:
                        print(f"Deduplicated cached results: {initial_count} -> {final_count} rows")
            else:
                if verbose:
                    print("Warning: Could not deduplicate cached results due to missing key columns.")
                # If key columns are missing, we can't reliably deduplicate.
                # We'll proceed, and if existing_results is still populated,
                # the later check for tokenizer_hash might catch issues.
                # If it becomes empty, load_model_results will be tried.

            if existing_results.empty and verbose:
                 print(f"Cached results became empty after deduplication/check or were initially empty.")
            elif verbose and not all(col in existing_results.columns for col in id_columns):
                 print(f"Proceeding with cached results, but key columns were missing for full deduplication.")
            elif verbose:
                print(f"Found and deduplicated cached results with matching tokenizer hash.")

        # TODO: Deal with tokenizer hash mismatch
        # If not found in cache (or cache was empty/problematic after dedupe), try loading the specific file
        if existing_results.empty:
            if verbose:
                print("Attempting to load specific model results file as cache was empty or problematic...")
            existing_results = load_model_results(model_name, revision, subfolder, _target_mode, results_dir)
            if not existing_results.empty:
                # Check if the loaded specific file has the correct tokenizer hash
                if tokenizer_hash_field in existing_results.columns:
                    existing_tokenizer_hash = existing_results[tokenizer_hash_field].iloc[0]
                    if existing_tokenizer_hash != tokenizer_hash:
                        if verbose:
                            print(f"Warning: Tokenizer hash mismatch in loaded file: {existing_tokenizer_hash} vs current {tokenizer_hash}.")
                    elif verbose:
                        print(f"Loaded specific results file with matching tokenizer hash")
                else:
                     if verbose:
                        print(f"Warning: Loaded specific results file lacks '{tokenizer_hash_field}'.")
            elif verbose:
                 print("No specific results file found or it was empty.")

    return existing_results

# Combine existing and new results
def combine_results(existing_results, new_results, files_to_evaluate, tokenizer_hash):
    """Combine existing and new results."""
    tokenizer_hash_field = get_tokenizer_hash_field()

    columns_to_keep = ALL_COLUMNS.copy()
    for column in PARITY_COLUMNS:
        if column in columns_to_keep:
            columns_to_keep.remove(column)

    if not new_results:
        # If no new evaluations, use existing results if they match the tokenizer hash
        if not existing_results.empty:
            result_df = existing_results[existing_results[tokenizer_hash_field] == tokenizer_hash]
            return ensure_dataframe_has_columns(result_df, ALL_COLUMNS)[columns_to_keep]
        return ensure_dataframe_has_columns(pd.DataFrame(), ALL_COLUMNS)[columns_to_keep]
    
    new_df = pd.DataFrame(new_results)
    new_df = ensure_dataframe_has_columns(new_df, ALL_COLUMNS)[columns_to_keep]
    
    if not existing_results.empty:
        # Keep only existing results with the current tokenizer hash
        existing_to_keep = existing_results[existing_results[tokenizer_hash_field] == tokenizer_hash]
        existing_to_keep = ensure_dataframe_has_columns(existing_to_keep, ALL_COLUMNS)[columns_to_keep]
        
        # Remove any entries that we've just re-evaluated
        for file_info in files_to_evaluate:
            lang = file_info['lang']
            domain = file_info['domain']
            existing_to_keep = existing_to_keep[
                ~((existing_to_keep['lang'] == lang) & (existing_to_keep['domain'] == domain))
            ]
        
        # Combine with new results
        if not existing_to_keep.empty:
            combined_df = pd.concat([existing_to_keep, new_df], ignore_index=True)
            return ensure_dataframe_has_columns(combined_df, ALL_COLUMNS)[columns_to_keep]
        else:
            return new_df
    else:
        return new_df

# Download the benchmark and results repositories
def extract_result_zips(evaluations_dir) -> dict:
    """
    Extract all batch_*.zip archives in the evaluations directory.
    Skips JSONL files that already exist on disk.

    Returns:
        dict mapping filename (str) -> zip_path (Path) for every JSONL member
        found across all batch zips (whether or not extraction was needed).
    """
    evaluations_dir = Path(evaluations_dir)
    zip_files = sorted(evaluations_dir.glob("batch_*.zip"))
    file_to_zip: dict = {}
    if not zip_files:
        return file_to_zip
    for zf_path in zip_files:
        with zipfile.ZipFile(zf_path, "r") as zf:
            for member in zf.namelist():
                if not member.endswith(".jsonl"):
                    continue
                file_to_zip[member] = zf_path
                dest = evaluations_dir / member
                if dest.exists():
                    continue
                zf.extract(member, evaluations_dir)
    print(f"Extracted {len(zip_files)} batch zip(s) in {evaluations_dir}", flush=True)
    return file_to_zip


def update_files_in_zip(zip_path: Path, updated_files: dict) -> None:
    """
    Replace specific members of a zip archive with updated local files.

    Args:
        zip_path: Path to the zip archive to update.
        updated_files: mapping of member_name (str) -> local_file_path (Path).
            All other members are preserved unchanged.
    """
    tmp_path = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(zip_path, "r") as zf_in:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
            for member in zf_in.namelist():
                if member in updated_files:
                    zf_out.write(updated_files[member], member)
                else:
                    zf_out.writestr(member, zf_in.read(member))
    tmp_path.replace(zip_path)


def download_repos(data_dir=DATA_DIR):
    """
    Download the benchmark and results repositories if not already present.
    
    Returns:
        benchmark_dir (Path): Local path to the benchmark dataset
        results_dir (Path): Local path to the results dataset
    """
    benchmark_dir = Path(data_dir) / DEFAULT_BENCHMARK_PATH
    results_dir = Path(data_dir) / DEFAULT_RESULTS_PATH
    
    # Download if not present
    local_benchmark = snapshot_download(
        repo_id=HF_REPO_BENCHMARK,
        local_dir=benchmark_dir,
        repo_type='dataset',
        max_workers=4,
    )
    local_results = snapshot_download(
       repo_id=HF_REPO_RESULTS,
       local_dir=results_dir,
       repo_type="dataset",
       allow_patterns=[
           f"{DEFAULT_MODEL_EVALUATIONS_PATH}/batch_*.zip",
           f"{DEFAULT_MODEL_EVALUATIONS_PATH}/*.jsonl",
           f"{DEFAULT_TOKENIZER_METADATA_PATH}/*.json",
       ],
       max_workers=4,
    )

    extract_result_zips(Path(local_results) / DEFAULT_MODEL_EVALUATIONS_PATH)
    
    return Path(local_benchmark), Path(local_results)

# Precalculate file hashes for all dataset files
def precalculate_file_hashes(benchmark_dir):
    """
    Precalculate file hashes for all dataset files.
    
    Returns:
        benchmark_file_hashes (dict): Dictionary of benchmark file paths to hash pairs
    """
    benchmark_file_hashes = {}
    
    # Load metadata to find all dataset files
    meta_path = Path(benchmark_dir) / 'dataset_meta.yaml'
    if meta_path.exists():
        meta = load_dataset_meta(meta_path)
        for lang, lang_info in meta.items():
            if not isinstance(lang_info, dict) or 'domains' not in lang_info:
                continue
            
            for domain in lang_info['domains']:
                file_path = Path(benchmark_dir) / domain['filepath']
                if file_path.exists():
                    # Calculate and store the file hash
                    benchmark_file_hashes[str(file_path)] = calculate_file_hash(file_path)
    
    return benchmark_file_hashes

# Identify which files need to be evaluated
def get_files_to_evaluate(meta, langs, benchmark_dir, benchmark_file_hashes, existing_results, tokenizer_hash, verbose=False):
    """Identify which files need to be evaluated."""
    # Filter languages if specified
    if langs:
        filtered_meta = {lang: meta[lang] for lang in langs if lang in meta}
    else:
        filtered_meta = meta
    
    # Track which files need to be evaluated
    files_to_evaluate = []
    added_filepaths = set()
    
    # Identify which files need to be evaluated
    for lang, lang_info in filtered_meta.items():
        if not isinstance(lang_info, dict) or 'domains' not in lang_info:
            continue

        extra_domains = []
        if 'parities' in lang_info and lang_info['parities'] and len(lang_info['parities']) > 0:
            for parity_data in lang_info['parities']:
                extra_domains.append({
                    'lang': lang,
                    'reference': False,
                    'name': parity_data['name'],
                    'filepath': parity_data['filepath'],
                    'evaluation_type': 'parity'
                })
                extra_domains.append({
                    'lang': parity_data['reference_lang'],
                    'reference': True,
                    'name': parity_data['name'],
                    'filepath': parity_data['reference_filepath'],
                    'evaluation_type': 'parity'
                })
        
        for domain in lang_info['domains'] + extra_domains:
            file_path = Path(benchmark_dir) / domain['filepath']
            if not file_path.exists():
                continue
            
            if file_path in added_filepaths:
                continue
            
            domain_name = domain['name']
            
            # Use precalculated hash if available, else calculate
            if benchmark_file_hashes and str(file_path) in benchmark_file_hashes:
                benchmark_file_hash = benchmark_file_hashes[str(file_path)]
                #if verbose:
                #    print(f"  Using precalculated hash for {domain_name}")
            else:
                # Calculate file hash
                benchmark_file_hash = calculate_file_hash(file_path)
            
            # Check if we already have results for this file with matching hashes
            need_evaluation = True
            if not existing_results.empty:
                # Find matching file and tokenizer hash
                matching_results = existing_results[
                    (existing_results['lang'] == domain.get('lang', lang)) & 
                    (existing_results['domain'] == domain_name) & 
                    # TODO: Deal with tokenizer hash mismatch
                    #(existing_results[get_tokenizer_hash_field()] == tokenizer_hash) &
                    (existing_results['benchmark_file_hash'] == benchmark_file_hash)
                ]
                
                if not matching_results.empty:
                    need_evaluation = False
                    if verbose:
                        print(f"  Skipping {lang}/{domain_name}: Already processed with matching hashes")
            
            if need_evaluation:
                files_to_evaluate.append({
                    'lang': domain.get('lang', lang),
                    'domain': domain_name,
                    'file_path': file_path,
                    'benchmark_file_hash': benchmark_file_hash,
                    'evaluation_type': domain.get('evaluation_type', 'standard'),
                    'reference': domain.get('reference', False)
                })
            added_filepaths.add(file_path)
    
    return files_to_evaluate

# Utility to download required repos only once
def download_repos_once():
    """
    Download the benchmark and results repositories if not already present.
    Precalculate file hashes for all dataset files.
    
    Returns:
        benchmark_dir (Path): Local path to the benchmark dataset
        results_dir (Path): Local path to the results dataset
        benchmark_file_hashes (dict): Dictionary of benchmark file paths to hash pairs
        tokenizer_hash_cache (dict): Dictionary mapping tokenizer hashes to existing results
    """
    benchmark_dir, results_dir = download_repos()
    
    # Precalculate file hashes
    benchmark_file_hashes = precalculate_file_hashes(benchmark_dir)
    
    # Build the tokenizer hash cache
    tokenizer_hash_cache = build_tokenizer_hash_cache(results_dir)
    
    return benchmark_dir, results_dir, benchmark_file_hashes, tokenizer_hash_cache

def _compute_language_rows(
    tokenizer,
    meta,
    benchmark_dir,
    affected_langs: List[str],
    fresh_lang_counters: Dict[str, Counter],
    fresh_file_keys: set,
    common_meta: dict,
    domain_df: "pd.DataFrame",
    benchmark_file_hashes: Optional[dict] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    _target_mode: bool = False,
    verbose: bool = False,
) -> List[dict]:
    """
    Compute per-language rows (evaluation_type='language') for affected languages.

    Standard metrics (fertility, compression, reversible_ratio, etc.) are derived by
    summing raw totals from the per-domain rows in domain_df. Zipf metrics require token
    frequency Counters: fresh domains already have them in fresh_lang_counters; cached
    domains are re-tokenized with collect_frequencies_batched.

    Args:
        tokenizer: The loaded tokenizer.
        meta: Dataset metadata dict (lang -> lang_info with 'domains' list).
        benchmark_dir: Path to the benchmark directory.
        affected_langs: Languages that had at least one domain freshly evaluated.
        fresh_lang_counters: Dict lang -> Counter accumulated during fresh evaluation.
        fresh_file_keys: Set of (lang, domain) pairs that were freshly evaluated.
        common_meta: Metadata dict with model/tokenizer fields to attach to each row.
        domain_df: Combined DataFrame containing all standard domain rows for this model.
        benchmark_file_hashes: Optional pre-computed file hash dict.
        batch_size: Tokenization batch size for cached domains.
        _target_mode: Whether to use target-mode tokenization.
        verbose: Verbose logging.

    Returns:
        Tuple of (list of result dicts with evaluation_type='language',
                  dict mapping lang -> merged Counter for that language).
    """
    language_rows = []
    lang_counters_out: Dict[str, Counter] = {}

    for lang in affected_langs:
        if lang not in meta or not isinstance(meta[lang], dict):
            continue
        lang_info = meta[lang]
        domains = lang_info.get('domains', [])

        # Find all standard (non-parity) domain files for this language
        all_domain_files = []
        for domain in domains:
            file_path = Path(benchmark_dir) / domain['filepath']
            if not file_path.exists():
                continue
            domain_name = domain['name']
            fhash = (benchmark_file_hashes or {}).get(str(file_path)) or calculate_file_hash(file_path)
            all_domain_files.append({
                'lang': lang,
                'domain': domain_name,
                'file_path': file_path,
                'benchmark_file_hash': fhash,
                'evaluation_type': 'standard',
                'reference': False,
            })

        if not all_domain_files:
            continue

        # Separate cached vs fresh domains for this language
        cached_files = [
            f for f in all_domain_files
            if (f['lang'], f['domain']) not in fresh_file_keys
        ]

        # Start with frequencies from fresh evaluation
        merged_counter: Counter = Counter(fresh_lang_counters.get(lang, Counter()))

        # Re-tokenize cached domains to get their frequencies
        if cached_files:
            if verbose:
                print(f"  Collecting frequencies for {len(cached_files)} cached domains for lang={lang}")
            cached_counters = collect_frequencies_batched(
                tokenizer, cached_files, batch_size=batch_size, _target_mode=_target_mode
            )
            merged_counter += cached_counters.get(lang, Counter())

        if not merged_counter:
            continue

        lang_counters_out[lang] = merged_counter

        zipf_metrics = compute_zipf_metrics(merged_counter, vocab_size=common_meta.get('vocab_size'))
        evaluation_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Sum raw totals from per-domain rows in domain_df
        lang_domain_rows = domain_df[
            (domain_df['lang'] == lang) &
            (domain_df['evaluation_type'] == 'standard') &
            (~domain_df['reference'].fillna(False))
        ]
        total_tokens_sum                = int(lang_domain_rows['total_tokens'].sum())
        total_words_sum                 = int(lang_domain_rows['total_words'].sum())
        total_chars_sum                 = int(lang_domain_rows['total_chars'].sum())
        total_bytes_sum                 = int(lang_domain_rows['total_bytes'].sum())
        total_docs_sum                  = int(lang_domain_rows['total_docs'].sum())
        reversible_docs_count_sum       = int(lang_domain_rows['reversible_docs_count'].sum())
        unk_tokens_count_sum            = int(lang_domain_rows['unk_tokens_count'].sum())
        byte_edit_distance_total_sum    = int(lang_domain_rows['byte_edit_distance_total'].sum())
        byte_edit_denominator_total_sum = int(lang_domain_rows['byte_edit_denominator_total'].sum())
        decoded_bytes_total_sum         = int(lang_domain_rows['decoded_bytes_total'].sum())
        char_edit_distance_total_sum    = int(lang_domain_rows['char_edit_distance_total'].sum())
        char_edit_denominator_total_sum = int(lang_domain_rows['char_edit_denominator_total'].sum())
        decoded_chars_total_sum         = int(lang_domain_rows['decoded_chars_total'].sum())

        compression_bytes_val      = calculate_compression(total_bytes_sum, total_tokens_sum)
        fertility_val              = calculate_fertility(total_tokens_sum, total_words_sum)
        compression_chars_val      = calculate_compression(total_chars_sum, total_tokens_sum)
        reversible_ratio_val       = calculate_reversible_ratio(reversible_docs_count_sum, total_docs_sum)
        unk_ratio_val              = calculate_unk_ratio(unk_tokens_count_sum, total_tokens_sum)
        byte_fidelity_val          = calculate_byte_fidelity(byte_edit_distance_total_sum, byte_edit_denominator_total_sum)
        effective_bytes_per_token_val = calculate_effective_bytes_per_token(compression_bytes_val, byte_fidelity_val)
        ebpb_val                   = calculate_ebpb(total_tokens_sum, total_bytes_sum, byte_edit_distance_total_sum, zipf_metrics.get('freq_cardinality'))
        ebpb_full_val              = calculate_ebpb_full(total_tokens_sum, total_bytes_sum, byte_edit_distance_total_sum, common_meta.get('vocab_size'))
        char_fidelity_val          = calculate_char_fidelity(char_edit_distance_total_sum, char_edit_denominator_total_sum)
        effective_chars_per_token_val = calculate_effective_chars_per_token(compression_chars_val, char_fidelity_val)
        ebpc_val                   = calculate_ebpc(common_meta.get('vocab_size'), effective_chars_per_token_val)

        row = {
            **common_meta,
            "lang": lang,
            "domain": "lang_overall",
            "evaluation_type": "language",
            "reference": False,
            "benchmark_file_hash": "",
            "evaluation_date": evaluation_date,
            "fertility": fertility_val,
            "compression_chars": compression_chars_val,
            "compression_bytes": compression_bytes_val,
            "reversible_ratio": reversible_ratio_val,
            "unk_ratio": unk_ratio_val,
            "byte_fidelity": byte_fidelity_val,
            "effective_bytes_per_token": effective_bytes_per_token_val,
            "ebpb": ebpb_val,
            "ebpb_full": ebpb_full_val,
            "char_fidelity": char_fidelity_val,
            "effective_chars_per_token": effective_chars_per_token_val,
            "ebpc": ebpc_val,
            "total_tokens": total_tokens_sum,
            "total_words": total_words_sum,
            "total_chars": total_chars_sum,
            "total_bytes": total_bytes_sum,
            "total_docs": total_docs_sum,
            "reversible_docs_count": reversible_docs_count_sum,
            "unk_tokens_count": unk_tokens_count_sum,
            "byte_edit_distance_total": byte_edit_distance_total_sum,
            "byte_edit_denominator_total": byte_edit_denominator_total_sum,
            "decoded_bytes_total": decoded_bytes_total_sum,
            "char_edit_distance_total": char_edit_distance_total_sum,
            "char_edit_denominator_total": char_edit_denominator_total_sum,
            "decoded_chars_total": decoded_chars_total_sum,
            **zipf_metrics,
        }
        language_rows.append(row)

        if verbose:
            print(f"  Language row for {lang}: {zipf_metrics}")

    return language_rows, lang_counters_out


def _compute_multilingual_row(
    tokenizer,
    meta,
    benchmark_dir,
    lang_counters: Dict[str, Counter],
    common_meta: dict,
    domain_df: "pd.DataFrame",
    benchmark_file_hashes: Optional[dict] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    _target_mode: bool = False,
    verbose: bool = False,
) -> Optional[dict]:
    """
    Compute a single evaluation_type='multilingual' row for a model when all languages
    are present.

    Zipf metrics are computed from a global Counter merged across all languages.
    Languages already in lang_counters (from _compute_language_rows) are used as-is;
    remaining languages are re-tokenized via collect_frequencies_batched.
    Standard metrics are derived by summing all standard, non-reference domain rows
    in domain_df.

    Args:
        tokenizer: The loaded tokenizer.
        meta: Full dataset metadata dict (lang -> lang_info).
        benchmark_dir: Path to the benchmark directory.
        lang_counters: Dict lang -> merged Counter from _compute_language_rows (affected langs only).
        common_meta: Metadata dict with model/tokenizer fields to attach to the row.
        domain_df: Combined DataFrame of all standard domain rows for this model.
        benchmark_file_hashes: Optional pre-computed file hash dict.
        batch_size: Tokenization batch size for missing languages.
        _target_mode: Whether to use target-mode tokenization.
        verbose: Verbose logging.

    Returns:
        A single result dict with evaluation_type='multilingual', or None on failure.
    """
    global_counter: Counter = Counter()

    # Merge counters from already-processed languages
    for lang_counter in lang_counters.values():
        global_counter += lang_counter

    # Collect frequencies for any language not already in lang_counters
    missing_langs = [lang for lang in meta if lang not in lang_counters]
    if missing_langs:
        if verbose:
            print(f"  Collecting frequencies for {len(missing_langs)} unprocessed languages for multilingual row")
        missing_files = []
        for lang in missing_langs:
            lang_info = meta.get(lang)
            if not isinstance(lang_info, dict):
                continue
            for domain in lang_info.get('domains', []):
                file_path = Path(benchmark_dir) / domain['filepath']
                if not file_path.exists():
                    continue
                fhash = (benchmark_file_hashes or {}).get(str(file_path)) or calculate_file_hash(file_path)
                missing_files.append({
                    'lang': lang,
                    'domain': domain['name'],
                    'file_path': file_path,
                    'benchmark_file_hash': fhash,
                    'evaluation_type': 'standard',
                    'reference': False,
                })
        if missing_files:
            missing_counters = collect_frequencies_batched(
                tokenizer, missing_files, batch_size=batch_size, _target_mode=_target_mode
            )
            for lang_counter in missing_counters.values():
                global_counter += lang_counter

    if not global_counter:
        return None

    zipf_metrics = compute_zipf_metrics(global_counter, vocab_size=common_meta.get('vocab_size'))
    # Sum raw totals from all standard, non-reference domain rows
    all_domain_rows = domain_df[
        (domain_df['evaluation_type'] == 'standard') &
        (~domain_df['reference'].fillna(False))
    ]
    total_tokens_sum                = int(all_domain_rows['total_tokens'].sum())
    total_words_sum                 = int(all_domain_rows['total_words'].sum())
    total_chars_sum                 = int(all_domain_rows['total_chars'].sum())
    total_bytes_sum                 = int(all_domain_rows['total_bytes'].sum())
    total_docs_sum                  = int(all_domain_rows['total_docs'].sum())
    reversible_docs_count_sum       = int(all_domain_rows['reversible_docs_count'].sum())
    unk_tokens_count_sum            = int(all_domain_rows['unk_tokens_count'].sum())
    byte_edit_distance_total_sum    = int(all_domain_rows['byte_edit_distance_total'].sum())
    byte_edit_denominator_total_sum = int(all_domain_rows['byte_edit_denominator_total'].sum())
    decoded_bytes_total_sum         = int(all_domain_rows['decoded_bytes_total'].sum())
    char_edit_distance_total_sum    = int(all_domain_rows['char_edit_distance_total'].sum())
    char_edit_denominator_total_sum = int(all_domain_rows['char_edit_denominator_total'].sum())
    decoded_chars_total_sum         = int(all_domain_rows['decoded_chars_total'].sum())

    compression_bytes_val         = calculate_compression(total_bytes_sum, total_tokens_sum)
    fertility_val                 = calculate_fertility(total_tokens_sum, total_words_sum)
    compression_chars_val         = calculate_compression(total_chars_sum, total_tokens_sum)
    reversible_ratio_val          = calculate_reversible_ratio(reversible_docs_count_sum, total_docs_sum)
    unk_ratio_val                 = calculate_unk_ratio(unk_tokens_count_sum, total_tokens_sum)
    byte_fidelity_val             = calculate_byte_fidelity(byte_edit_distance_total_sum, byte_edit_denominator_total_sum)
    effective_bytes_per_token_val = calculate_effective_bytes_per_token(compression_bytes_val, byte_fidelity_val)
    ebpb_val                      = calculate_ebpb(total_tokens_sum, total_bytes_sum, byte_edit_distance_total_sum, zipf_metrics.get('freq_cardinality'))
    ebpb_full_val                 = calculate_ebpb_full(total_tokens_sum, total_bytes_sum, byte_edit_distance_total_sum, common_meta.get('vocab_size'))
    char_fidelity_val             = calculate_char_fidelity(char_edit_distance_total_sum, char_edit_denominator_total_sum)
    effective_chars_per_token_val = calculate_effective_chars_per_token(compression_chars_val, char_fidelity_val)
    ebpc_val                      = calculate_ebpc(common_meta.get('vocab_size'), effective_chars_per_token_val)

    evaluation_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

    row = {
        **common_meta,
        "lang": "all",
        "domain": "multilingual_overall",
        "evaluation_type": "multilingual",
        "reference": False,
        "benchmark_file_hash": "",
        "evaluation_date": evaluation_date,
        "fertility": fertility_val,
        "compression_chars": compression_chars_val,
        "compression_bytes": compression_bytes_val,
        "reversible_ratio": reversible_ratio_val,
        "unk_ratio": unk_ratio_val,
        "byte_fidelity": byte_fidelity_val,
        "effective_bytes_per_token": effective_bytes_per_token_val,
        "ebpb": ebpb_val,
        "ebpb_full": ebpb_full_val,
        "char_fidelity": char_fidelity_val,
        "effective_chars_per_token": effective_chars_per_token_val,
        "ebpc": ebpc_val,
        "total_tokens": total_tokens_sum,
        "total_words": total_words_sum,
        "total_chars": total_chars_sum,
        "total_bytes": total_bytes_sum,
        "total_docs": total_docs_sum,
        "reversible_docs_count": reversible_docs_count_sum,
        "unk_tokens_count": unk_tokens_count_sum,
        "byte_edit_distance_total": byte_edit_distance_total_sum,
        "byte_edit_denominator_total": byte_edit_denominator_total_sum,
        "decoded_bytes_total": decoded_bytes_total_sum,
        "char_edit_distance_total": char_edit_distance_total_sum,
        "char_edit_denominator_total": char_edit_denominator_total_sum,
        "decoded_chars_total": decoded_chars_total_sum,
        **zipf_metrics,
    }

    if verbose:
        print(f"  Multilingual row: {zipf_metrics}")

    return row


# Main evaluation function
def run_benchmark(model_name, benchmark_dir, langs=None, revision="main", subfolder=None, _target_mode=False, verbose=False, save_results=True, force_rerun=False, benchmark_file_hashes=None, tokenizer_hash_cache=None, upload_results: bool = False, trust_remote_code: bool = False, output_path: str | Path | None = None, use_batched_evaluation=True, batch_size=DEFAULT_BATCH_SIZE):
    """
    Benchmark a tokenizer on the dataset.
    
    Args:
        model_name: HuggingFace model name or path
        benchmark_dir: Local path to the benchmark dataset
        langs: Optional list of languages to test (e.g., ['en', 'pt']). If None, test all languages.
        revision: Model revision to use (default: 'main')
        subfolder: Optional subfolder within the model repository
        _target_mode: Whether to enable target mode (default: False)
        verbose: If True, print progress information
        save_results: If True, save results to disk and upload to HF hub
        force_rerun: If True, force re-evaluation even if results exist
        benchmark_file_hashes: Optional dictionary of precalculated benchmark file hashes to avoid recalculation
        tokenizer_hash_cache: Optional dictionary mapping tokenizer hashes to existing results
        upload_results: If True, upload results to HF hub (requires save_results=True)
        trust_remote_code: If True, trust remote code when loading tokenizer.
        output_path: Optional path to copy the final results file to.
        use_batched_evaluation: If True, use batched evaluation for better efficiency. If False, use original single-file evaluation.
        batch_size: Maximum number of texts to process in each batch (only used if use_batched_evaluation=True).
        
    Returns:
        pandas DataFrame with results
    """
    general_time = time.time()
    results_dir = Path(DATA_DIR) / DEFAULT_RESULTS_PATH
    
    if verbose:
        print(f"Testing tokenizer: {model_name}")
        if langs:
            print(f"Language subset: {', '.join(langs)}")
        else:
            print("Testing on all available languages")
    
    dataset_dir = Path(benchmark_dir)
    meta_path = dataset_dir / 'dataset_meta.yaml'
    if not meta_path.exists():
        raise FileNotFoundError(f"dataset_meta.yaml not found in {dataset_dir}")
    
    # Load metadata and tokenizer
    meta = load_dataset_meta(meta_path)

    kwargs = {
        'trust_remote_code': trust_remote_code,
        'revision': revision,
    }
    if subfolder:
        kwargs['subfolder'] = subfolder
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, **kwargs
        )
    except Exception as e:
        print(f"Error loading tokenizer {model_name} with use_fast=True, trying without it: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, **kwargs
        )
    
    # Apply target mode if requested (for tokenizers like Marian/mBART/M2M100 with separate source/target vocabs)
    if _target_mode:
        if hasattr(tokenizer, '_switch_to_target_mode'):
            try:
                if verbose:
                    print(f"Switching tokenizer to target mode...")
                tokenizer._switch_to_target_mode()
                if verbose:
                    print(f"Successfully switched tokenizer to target mode")
            except Exception as e:
                raise RuntimeError(f"Failed to switch tokenizer to target mode: {type(e).__name__}: {e}")
        elif hasattr(tokenizer, 'as_target_tokenizer'):
            try:
                if verbose:
                    print(f"Switching tokenizer to target mode via as_target_tokenizer...")
                tokenizer.as_target_tokenizer()
                if verbose:
                    print(f"Successfully switched tokenizer to target mode")
            except Exception as e:
                raise RuntimeError(f"Failed to switch tokenizer to target mode: {type(e).__name__}: {e}")
        elif verbose:
            print(f"Tokenizer does not support target mode switching; using text_target= parameter instead")
    
    vocab_size = tokenizer.vocab_size
    tokenizer_class = tokenizer.__class__.__name__

    try:
        #model_info_data = api.model_info(model_name, expand=['sha', 'likes', 'downloads', 'downloadsAllTime', 'createdAt', 'lastModified', 'tags'])
        model_info_data = api.model_info(model_name, expand=['sha', 'likes', 'downloadsAllTime', 'createdAt'])
        likes = model_info_data.likes
        sha = getattr(model_info_data, 'sha', None)
        created_at = getattr(model_info_data, 'created_at', None)
        downloads_all_time = getattr(model_info_data, 'downloads_all_time', 0)
    except Exception as e:
        if verbose:
            print(f"Could not fetch model info for {model_name}: {e}")
        likes = 0
        sha = None
        created_at = None
        downloads_all_time = 0

    if created_at is not None and not isinstance(created_at, str):
        created_at = created_at.isoformat()

    author_info = get_author_info(api, model_name)

    # Calculate vocab hash
    vocab_hash, filtered_vocab = calculate_vocab_hash(tokenizer)
    if verbose:
        print(f"vocab hash: {vocab_hash}")

    tokenizer_config, tokenizer_config_hash = get_tokenizer_config(tokenizer, vocab_hash)
    if verbose:
        print(f"tokenizer config hash: {tokenizer_config_hash}")

    assert TOKENIZER_HASH_FIELD in ['tokenizer_config_hash', 'vocab_hash']
    tokenizer_hash = tokenizer_config_hash if TOKENIZER_HASH_FIELD == 'tokenizer_config_hash' else vocab_hash # it was vocab_hash, but we now use tokenizer_config_hash
        
    # Check for existing results unless forced to rerun
    existing_results = pd.DataFrame()
    if not force_rerun:
        existing_results = get_model_results(tokenizer_hash, model_name, revision, subfolder, _target_mode, force_rerun, verbose, results_dir, tokenizer_hash_cache)

        existing_results['likes'] = likes
        existing_results['sha'] = sha
        existing_results['created_at'] = created_at
        existing_results['downloads_all_time'] = downloads_all_time
        existing_results['author_type'] = author_info.get('author_type')
        existing_results['author_fullname'] = author_info.get('author_fullname')
        existing_results['author_followers'] = author_info.get('author_followers', 0)
        existing_results['author_models'] = author_info.get('author_models', 0)
        existing_results['author_is_verified'] = author_info.get('author_is_verified', False)

    # Get files that need evaluation
    files_to_evaluate = get_files_to_evaluate(
        meta, langs, benchmark_dir, benchmark_file_hashes, existing_results, tokenizer_hash, verbose
    )
    
    # Perform evaluations only on files that need it
    new_results = []
    
    if verbose and files_to_evaluate:
        print(f"Evaluating {len(files_to_evaluate)} new or changed files")
    elif verbose:
        print("No new files to evaluate")

    near_duplicates = None
    vocab_near_duplicates = None
    vocab_prefix = None
    vocab_count_duplicates_by_type = None
    tokenizer_meta = None
    if len(files_to_evaluate) > 0:
        # calculate near duplicates
        near_duplicates = None
        if filtered_vocab:
            #print("Calculating vocabulary Near-Duplicates")
            near_duplicates = detect_near_duplicated_tokens(filtered_vocab, verbose=verbose)
            vocab_prefix = near_duplicates.get('vocab_prefix')

            vocab_count_duplicates_by_type = near_duplicates.get('count_by_step')
            vocab_near_duplicates = near_duplicates.get(DUPLICATE_MAIN_FIELD)

            if verbose:
                print(f"Tokenizer vocab near_duplicates: {vocab_near_duplicates*100:.2f}%")

        tokenizer_meta = build_tokenizer_meta(tokenizer, model_name, vocab_hash, tokenizer_config_hash, tokenizer_config, near_duplicates, filtered_vocab, revision, subfolder, _target_mode)
    
    if use_batched_evaluation and files_to_evaluate:
        # Use batched evaluation for better efficiency
        if verbose:
            print(f"Using batched evaluation strategy with batch_size={batch_size}")
        
        batch_results, fresh_lang_counters = evaluate_lang_domains_batched(
            tokenizer, files_to_evaluate, batch_size=batch_size, _target_mode=_target_mode, verbose=verbose
        )
        
        # Add common metadata to all results
        for res in batch_results:
            res.update({
                "model": model_name,
                "revision": revision,
                "subfolder": subfolder,
                "_target_mode": _target_mode,
                "model_key": generate_model_key(model_name, revision, subfolder, _target_mode),
                "vocab_size": vocab_size,
                "likes": likes,
                "sha": sha,
                "created_at": created_at,
                "downloads_all_time": downloads_all_time,
                "vocab_prefix": vocab_prefix,
                "vocab_near_duplicates": vocab_near_duplicates,
                "vocab_count_duplicates_by_type": vocab_count_duplicates_by_type,
                "tokenizer_class": tokenizer_class,
                "tokenizer_algorithm": tokenizer_meta.get("tokenizer_algorithm"),
                "tokenizer_implementation": tokenizer_meta.get("tokenizer_implementation"),
                "tokenizer_config_hash": tokenizer_meta.get("tokenizer_config_hash"),
                #"tokenizer_config": tokenizer_meta.get("tokenizer_config"),
                "vocab_hash": vocab_hash,
                "reference": False,  # Default, will be updated below if needed
                "evaluation_type": "standard",  # Default, will be updated below if needed
                "author_type": author_info.get('author_type'),
                "author_fullname": author_info.get('author_fullname'),
                "author_followers": author_info.get('author_followers', 0),
                "author_models": author_info.get('author_models', 0),
                "author_is_verified": author_info.get('author_is_verified', False),
            })
        
        # Update specific fields from file_info
        result_lookup = {(res['lang'], res['domain']): res for res in batch_results}
        for file_info in files_to_evaluate:
            key = (file_info['lang'], file_info['domain'])
            if key in result_lookup:
                result_lookup[key]["reference"] = file_info.get('reference', False)
                result_lookup[key]["evaluation_type"] = file_info.get('evaluation_type', 'standard')
        
        new_results.extend(batch_results)
     
    elif files_to_evaluate:
        # Use original single-file evaluation
        if verbose:
            print("Using original single-file evaluation strategy")
        
        fresh_lang_counters: Dict[str, Counter] = {}

        for file_info in files_to_evaluate:
            lang = file_info['lang']
            domain_name = file_info['domain'] 
            file_path = file_info['file_path']
            benchmark_file_hash = file_info['benchmark_file_hash']
            evaluation_type = file_info['evaluation_type']
            reference = file_info['reference']
            
            if verbose:
                print(f"Processing language: {lang}")
            
            meta_info = {
                'lang': lang,
                'domain': domain_name,
            }
            
            res, domain_counter = evaluate_lang_domain(tokenizer, file_path, meta_info, _target_mode=_target_mode, verbose=verbose)
            # Accumulate counter for this language
            if evaluation_type == 'standard' and not reference:
                if lang not in fresh_lang_counters:
                    fresh_lang_counters[lang] = Counter()
                fresh_lang_counters[lang] += domain_counter

            res.update({
                "model": model_name,
                "revision": revision,
                "subfolder": subfolder,
                "_target_mode": _target_mode,
                "model_key": generate_model_key(model_name, revision, subfolder, _target_mode),
                "vocab_size": vocab_size,
                "likes": likes,
                "sha": sha,
                "created_at": created_at,
                "downloads_all_time": downloads_all_time,
                "vocab_prefix": vocab_prefix,
                "vocab_near_duplicates": vocab_near_duplicates,
                "vocab_count_duplicates_by_type": vocab_count_duplicates_by_type,
                "tokenizer_class": tokenizer_class,
                "tokenizer_algorithm": tokenizer_meta.get("tokenizer_algorithm"),
                "tokenizer_implementation": tokenizer_meta.get("tokenizer_implementation"),
                "tokenizer_config_hash": tokenizer_meta.get("tokenizer_config_hash"),
                #"tokenizer_config": tokenizer_meta.get("tokenizer_config"),
                "vocab_hash": vocab_hash,
                "benchmark_file_hash": benchmark_file_hash,
                "reference": reference,
                "evaluation_type": evaluation_type,
                "author_type": author_info.get('author_type'),
                "author_fullname": author_info.get('author_fullname'),
                "author_followers": author_info.get('author_followers', 0),
                "author_models": author_info.get('author_models', 0),
                "author_is_verified": author_info.get('author_is_verified', False),
            })
            new_results.append(res)
    else:
        fresh_lang_counters: Dict[str, Counter] = {}

    # Compute per-language Zipf rows for any language that had fresh domain evaluations
    fresh_file_keys = {
        (f['lang'], f['domain'])
        for f in files_to_evaluate
        if f.get('evaluation_type', 'standard') == 'standard' and not f.get('reference', False)
    }
    affected_langs = list({
        lang for lang, _domain in fresh_file_keys
    })

    if affected_langs and tokenizer_meta is not None:
        common_meta = {
            "model": model_name,
            "revision": revision,
            "subfolder": subfolder,
            "_target_mode": _target_mode,
            "model_key": generate_model_key(model_name, revision, subfolder, _target_mode),
            "vocab_size": vocab_size,
            "likes": likes,
            "sha": sha,
            "created_at": created_at,
            "downloads_all_time": downloads_all_time,
            "vocab_prefix": vocab_prefix,
            "vocab_near_duplicates": vocab_near_duplicates,
            "vocab_count_duplicates_by_type": vocab_count_duplicates_by_type,
            "tokenizer_class": tokenizer_class,
            "tokenizer_algorithm": tokenizer_meta.get("tokenizer_algorithm"),
            "tokenizer_implementation": tokenizer_meta.get("tokenizer_implementation"),
            "tokenizer_config_hash": tokenizer_meta.get("tokenizer_config_hash"),
            "vocab_hash": vocab_hash,
            "author_type": author_info.get('author_type'),
            "author_fullname": author_info.get('author_fullname'),
            "author_followers": author_info.get('author_followers', 0),
            "author_models": author_info.get('author_models', 0),
            "author_is_verified": author_info.get('author_is_verified', False),
        }
    print(f"New results: {len(new_results)}")
    # Combine existing results (with matching tokenizer hash) and new results
    df = combine_results(existing_results, new_results, files_to_evaluate, tokenizer_hash)

    if affected_langs and tokenizer_meta is not None:
        language_rows, lang_counters = _compute_language_rows(
            tokenizer=tokenizer,
            meta=meta,
            benchmark_dir=benchmark_dir,
            affected_langs=affected_langs,
            fresh_lang_counters=fresh_lang_counters,
            fresh_file_keys=fresh_file_keys,
            common_meta=common_meta,
            domain_df=df,
            benchmark_file_hashes=benchmark_file_hashes,
            batch_size=batch_size,
            _target_mode=_target_mode,
            verbose=verbose,
        )
        if language_rows:
            df = df[~(
                (df['evaluation_type'] == 'language') &
                (df['lang'].isin(affected_langs))
            )]
            df = pd.concat([df, pd.DataFrame(language_rows)], ignore_index=True)

        # Compute multilingual row if all languages are now present
        all_meta_langs = set(meta.keys())
        evaluated_langs = set(
            df[df['evaluation_type'] == 'standard']['lang'].unique()
        )
        if all_meta_langs.issubset(evaluated_langs):
            multilingual_row = _compute_multilingual_row(
                tokenizer=tokenizer,
                meta=meta,
                benchmark_dir=benchmark_dir,
                lang_counters=lang_counters,
                common_meta=common_meta,
                domain_df=df,
                benchmark_file_hashes=benchmark_file_hashes,
                batch_size=batch_size,
                _target_mode=_target_mode,
                verbose=verbose,
            )
            if multilingual_row:
                df = df[df['evaluation_type'] != 'multilingual']
                df = pd.concat([df, pd.DataFrame([multilingual_row])], ignore_index=True)

        # Save results if requested
    if save_results and not df.empty:# and new_results:  # Only save if we have new results
        save_model_results(df, model_name, revision=revision, subfolder=subfolder, _target_mode=_target_mode, results_dir=results_dir, upload_to_hub=upload_results, output_path=output_path, verbose=verbose)
        # save_tokenizer_metadata(
        #     tokenizer_name=model_name,
        #     tokenizer=tokenizer,
        #     vocab_hash=vocab_hash,
        #     near_duplicates=near_duplicates,
        #     filtered_vocab=filtered_vocab,
        #     results_dir=results_dir,
        # )
    
    # Calculate and display overall results if verbose
    if verbose and not df.empty:
        overalls = calculate_overall_for_model(df, meta=meta)
        
        print("\nPer-language overall results:")
        for lang_overall in overalls.get("lang_overalls", []):
            lang = lang_overall["lang"]
            fertility = lang_overall["fertility"]
            compression_chars = lang_overall["compression_chars"]
            compression_bytes = lang_overall["compression_bytes"]
            reversible_ratio = lang_overall["reversible_ratio"]
            unk_ratio = lang_overall["unk_ratio"]
            byte_fidelity = lang_overall.get("byte_fidelity")
            effective_bytes_per_token = lang_overall.get("effective_bytes_per_token")
            ebpb = lang_overall.get("ebpb")
            tokens = lang_overall.get("total_tokens", "N/A")
            words = lang_overall.get("total_words", "N/A")
            chars = lang_overall.get("total_chars", "N/A")
            docs = lang_overall.get("total_docs", "N/A")
            bytes = lang_overall.get("total_bytes", "N/A")
            eval_date = lang_overall.get("evaluation_date", "N/A")
            
            char_fidelity = lang_overall.get("char_fidelity")
            effective_chars_per_token = lang_overall.get("effective_chars_per_token")
            ebpc = lang_overall.get("ebpc")
            print(f"  {lang}: Fertility: {fertility}, Compression (Chars): {compression_chars}, Compression (Bytes): {compression_bytes}, Reversible: {reversible_ratio}, Tokens: {tokens}, Words: {words}, Chars: {chars}, Docs: {docs}, Bytes: {bytes}, Date: {eval_date}, Unk Ratio: {unk_ratio}, Byte Fidelity: {byte_fidelity}, Effective Bytes/Token: {effective_bytes_per_token}, EBPB: {ebpb}, Char Fidelity: {char_fidelity}, Effective Chars/Token: {effective_chars_per_token}, EBPC: {ebpc}")
            freq_cardinality = lang_overall.get("freq_cardinality")
            freq_auc = lang_overall.get("freq_auc")
            freq_slope = lang_overall.get("freq_slope")
            freq_power_law = lang_overall.get("freq_power_law")
            print(f"  Zipf: Cardinality: {freq_cardinality}, AUC: {freq_auc}, Slope: {freq_slope}, Power Law Dev: {freq_power_law}")
            parity_columns = [col for col in lang_overall.keys() if col.startswith('parity_')]
            parity_text = ", ".join([f"{col}: {lang_overall[col]}" for col in parity_columns])
            print(f"  Parity: {parity_text}")

        if "multilingual_overall" in overalls:
            multi_fertility = overalls["multilingual_overall"]["fertility"]
            multi_compression_chars = overalls["multilingual_overall"]["compression_chars"]
            multi_compression_bytes = overalls["multilingual_overall"]["compression_bytes"]
            multi_reversible_ratio = overalls["multilingual_overall"]["reversible_ratio"]
            multi_unk_ratio = overalls["multilingual_overall"]["unk_ratio"]
            multi_byte_fidelity = overalls["multilingual_overall"].get("byte_fidelity")
            multi_effective_bytes_per_token = overalls["multilingual_overall"].get("effective_bytes_per_token")
            multi_ebpb = overalls["multilingual_overall"].get("ebpb")
            print(f"\nMultilingual overall fertility: {multi_fertility}")
            print(f"Multilingual overall compression (Chars): {multi_compression_chars}")
            print(f"Multilingual overall compression (Bytes): {multi_compression_bytes}")
            print(f"Multilingual overall reversible ratio: {multi_reversible_ratio}")
            print(f"Multilingual overall unk ratio: {multi_unk_ratio}")
            print(f"Multilingual overall byte fidelity: {multi_byte_fidelity}")
            print(f"Multilingual overall effective bytes/token: {multi_effective_bytes_per_token}")
            print(f"Multilingual overall EBPB: {multi_ebpb}")
            multi_char_fidelity = overalls["multilingual_overall"].get("char_fidelity")
            multi_effective_chars_per_token = overalls["multilingual_overall"].get("effective_chars_per_token")
            multi_ebpc = overalls["multilingual_overall"].get("ebpc")
            print(f"Multilingual overall char fidelity: {multi_char_fidelity}")
            print(f"Multilingual overall effective chars/token: {multi_effective_chars_per_token}")
            print(f"Multilingual overall EBPC: {multi_ebpc}")
            multi_freq_cardinality = overalls["multilingual_overall"].get("freq_cardinality")
            multi_freq_auc = overalls["multilingual_overall"].get("freq_auc")
            multi_freq_slope = overalls["multilingual_overall"].get("freq_slope")
            multi_freq_power_law = overalls["multilingual_overall"].get("freq_power_law")
            print(f"Multilingual Zipf: Cardinality: {multi_freq_cardinality}, AUC: {multi_freq_auc}, Slope: {multi_freq_slope}, Power Law Dev: {multi_freq_power_law}")
            if 'total_tokens' in overalls['multilingual_overall']:
                print(f"Total tokens: {overalls['multilingual_overall']['total_tokens']}")
            if 'total_words' in overalls['multilingual_overall']:
                print(f"Total words: {overalls['multilingual_overall']['total_words']}")
            if 'total_chars' in overalls['multilingual_overall']:
                print(f"Total chars: {overalls['multilingual_overall']['total_chars']}")
            if 'total_bytes' in overalls['multilingual_overall']:
                print(f"Total bytes: {overalls['multilingual_overall']['total_bytes']}")
            if 'total_docs' in overalls['multilingual_overall']:
                print(f"Total docs: {overalls['multilingual_overall']['total_docs']}")
            if 'evaluation_date' in overalls['multilingual_overall']:
                print(f"Last Eval Date: {overalls['multilingual_overall']['evaluation_date']}")

            parity_columns = [col for col in overalls['multilingual_overall'].keys() if col.startswith('parity_')]
            parity_text = ", ".join([f"{col}: {overalls['multilingual_overall'][col]}" for col in parity_columns])
            print(f"Multilingual Parity: {parity_text}")
    
    time_total = time.time() - general_time
    print(f"Run benchmark total time: {time_total:.2f} seconds")

    return df

# Note: get_overall_summary is now in results_aggregator module to avoid circular imports

def get_tokenizer_type(tokenizer: PreTrainedTokenizer) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristics to try to identify the tokenizer algorithm and underlying implementation/framework.

    Implementation:
        - Transformers: Tokenizers library
        - SentencePiece: SentencePiece library
        - Tokenizers: Tokenizers library
        - Tiktoken: Tiktoken library
    
    Algorithm:
        - BPE: Byte Pair Encoding
        - WordPiece: WordPiece
        - Unigram: Unigram
        - CharLevel: Character Level
        - WordLevel: Word Level
        - Unknown: Unknown

    Returns:
        A tuple of (algorithm, implementation).
    """

    implementation = 'Transformers'
    algorithm = 'Unknown'
    
    # 1. Fast tokenizers from `tokenizers` library
    if tokenizer.is_fast:
        implementation = 'Tokenizers'
        possible_model_files = []

        if os.path.isdir(tokenizer.name_or_path):
            spiece_or_tiktoken_model_files = ['.model', 'spm', 'spiece', 'sentencepiece', 'tiktoken', 'bpe']
            for file in os.listdir(tokenizer.name_or_path):
                if any(k in file.lower() for k in spiece_or_tiktoken_model_files):
                    possible_model_files.append(os.path.join(tokenizer.name_or_path, file))
        
        if os.path.isfile(tokenizer.name_or_path):
            possible_model_files.append(tokenizer.name_or_path)

        for filename in ['vocab_file', 'spm_file', 'spm_files']:
            if hasattr(tokenizer, filename) and \
                getattr(tokenizer, filename) is not None and \
                os.path.exists(getattr(tokenizer, filename)):

                model_file = getattr(tokenizer, filename)
                if isinstance(model_file, list):
                    model_file = model_file[0]

                possible_model_files.append(model_file)

            if hasattr(tokenizer, 'init_kwargs') and\
                filename in tokenizer.init_kwargs and \
                tokenizer.init_kwargs[filename] is not None and \
                os.path.exists(tokenizer.init_kwargs[filename]):

                model_file = tokenizer.init_kwargs[filename]
                if isinstance(model_file, list):
                    model_file = model_file[0]

                possible_model_files.append(model_file)
            
        for possible_model_file in possible_model_files:
            try:
                vocab_file = possible_model_file
                from transformers.convert_slow_tokenizer import import_protobuf

                proto = import_protobuf().ModelProto()
                with open(vocab_file, "rb") as f:
                    proto.ParseFromString(f.read())
                implementation = 'SentencePiece'
            except Exception as e:
                #print(e)
                try:
                    from tiktoken.load import load_tiktoken_bpe
                    load_tiktoken_bpe(possible_model_file)
                    implementation = 'Tiktoken'
                except Exception as e:
                    #print(e)
                    pass
            
            if implementation != 'Tokenizers':
                break

        try:
            algorithm = tokenizer.backend_tokenizer.model.__class__.__name__
            return algorithm, implementation
        except AttributeError:
            # Fallback for fast tokenizers if model attribute is not found
            pass

    # 2. Slow tokenizers
    # Check for SentencePiece
    if (hasattr(tokenizer, 'sp_model') and tokenizer.sp_model is not None) or hasattr(tokenizer, 'spm_file') or hasattr(tokenizer, 'spm_files'):
        implementation = 'SentencePiece'
        # Try to infer algorithm from vocab file name.
        vocab_file = None
        if hasattr(tokenizer, 'spm_files') and tokenizer.spm_files is not None:
            vocab_file = tokenizer.spm_files
        elif hasattr(tokenizer, 'spm_file') and tokenizer.spm_file is not None and os.path.exists(tokenizer.spm_file):
            vocab_file = tokenizer.spm_file
        elif hasattr(tokenizer, 'vocab_file') and tokenizer.vocab_file is not None and os.path.exists(tokenizer.vocab_file):
            vocab_file = tokenizer.vocab_file
        elif hasattr(tokenizer, 'init_kwargs') and 'vocab_file' in tokenizer.init_kwargs and tokenizer.init_kwargs['vocab_file'] is not None and os.path.exists(tokenizer.init_kwargs['vocab_file']):
            vocab_file = tokenizer.init_kwargs['vocab_file']

        if isinstance(vocab_file, list):
            vocab_file = vocab_file[0]

        if vocab_file is None:
            return 'Unknown', implementation

        try:
            from transformers.convert_slow_tokenizer import import_protobuf

            proto = import_protobuf().ModelProto()
            with open(vocab_file, "rb") as f:
                proto.ParseFromString(f.read())

            #https://github.com/google/sentencepiece/blob/273449044caa593c2fd7eb7550cb3ab2cff93f1a/src/sentencepiece_model.proto#L48
            model_type = proto.trainer_spec.model_type
            if model_type == 1:
                algorithm = 'Unigram'
            elif model_type == 2:
                algorithm = 'BPE'
            elif model_type == 3:
                algorithm = 'WordLevel'
            elif model_type == 4:
                algorithm = 'CharLevel'

            return algorithm, implementation
        except:
            traceback.print_exc()
            print('Error getting tokenizer type')
            pass
            
    if hasattr(tokenizer, 'tokenizer') and tokenizer.tokenizer is not None:
        if 'tiktoken' in type(tokenizer.tokenizer).__module__:
            implementation = 'Tiktoken'
            algorithm = 'BPE'
            return algorithm, implementation
    
    if hasattr(tokenizer, 'vocab_files_names') and tokenizer.vocab_files_names is not None:
        for _, value in tokenizer.vocab_files_names.items():
            value = value.lower()
            if 'tiktoken' in value:
                implementation = 'Tiktoken'
                algorithm = 'BPE'
                return algorithm, implementation
            if 'sentencepiece' in value or 'spm' in value or 'spiece' in value:
                implementation = 'SentencePiece'
                if 'bpe' in value:
                    algorithm = 'BPE'
                    return algorithm, implementation
            if 'bpe' in value:
                algorithm = 'BPE'
                
    # Check for other slow tokenizers
    prefix = detect_vocab_prefix(tokenizer.get_vocab())
    if prefix == 'Ġ':
        algorithm = 'BPE'
    elif prefix == "##":
        algorithm = 'WordPiece'
    elif prefix == chr(9601):
        implementation = 'SentencePiece'
    elif prefix == "</w>":
        algorithm = 'BPE'
        implementation = 'FastBPE'
    elif prefix == "@@":
        algorithm = 'BPE'
        implementation = 'FastBPE'
    
    if len(tokenizer.get_vocab()) < 2000:
        algorithm = 'CharLevel'

    return algorithm, implementation

# Note: Vectorized functions are now in results_aggregator module to avoid circular imports
# They can be imported directly from results_aggregator when needed

if __name__ == '__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Test tokenizer on a subset of languages')
    parser.add_argument('model_name', help='HuggingFace model name or path')
    parser.add_argument('--revision', default='main', help='HuggingFace model revision')
    parser.add_argument('--subfolder', default=None, help='Optional subfolder within the model repository')
    parser.add_argument('--target-mode', action='store_true', help='Enable target mode (calls _switch_to_target_mode on the tokenizer)')
    parser.add_argument('--langs', nargs='+', help='Language subsets to test (e.g., en pt code)', default=None)
    parser.add_argument('--no-save', action='store_true', help='Do not save results to disk')
    parser.add_argument('--no-force', action='store_true', help='Do not force re-evaluation, use existing results if available')
    parser.add_argument('--upload', action='store_true', help='Upload results to HF hub (requires --no-save to be false)')
    parser.add_argument('--trust-remote-code', action='store_true', help='Trust remote code')
    parser.add_argument('--output-path', help='Optional path to copy the results JSONL file to', default=None)
    parser.add_argument('--no-batched-eval', action='store_true', help='Disable batched evaluation (use original single-file strategy)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help=f'Batch size for batched evaluation (default: {DEFAULT_BATCH_SIZE})')
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    # Download required repos once
    force_rerun = not args.no_force
    if force_rerun:
        # If forcing re-evaluation, only download benchmark directory
        benchmark_dir = Path(DATA_DIR) / DEFAULT_BENCHMARK_PATH
        local_benchmark = snapshot_download(
            repo_id=HF_REPO_BENCHMARK,
            local_dir=benchmark_dir,
            repo_type='dataset'
        )
        benchmark_dir = Path(local_benchmark)
        benchmark_file_hashes = precalculate_file_hashes(benchmark_dir)
        tokenizer_hash_cache = None  # Don't need cache when forcing re-evaluation
    else:
        # Download both repos for comparison with existing results
        benchmark_dir, results_dir, benchmark_file_hashes, tokenizer_hash_cache = download_repos_once()
    
    # Run benchmark with specified options
    df = run_benchmark(
        model_name=args.model_name,
        revision=args.revision,
        subfolder=args.subfolder,
        _target_mode=args.target_mode,
        benchmark_dir=benchmark_dir,
        langs=args.langs,
        verbose=True,
        save_results=not args.no_save,
        force_rerun=force_rerun,
        benchmark_file_hashes=benchmark_file_hashes,
        tokenizer_hash_cache=tokenizer_hash_cache,
        upload_results=args.upload and not args.no_save, # Only upload if saving and requested
        trust_remote_code=args.trust_remote_code,
        output_path=args.output_path, # Pass the output file path
        use_batched_evaluation=not args.no_batched_eval,
        batch_size=args.batch_size
    )
    # Calculate overall metrics
    meta_path = Path(benchmark_dir) / 'dataset_meta.yaml'
    meta = load_dataset_meta(meta_path) if meta_path.exists() else None
    overalls = calculate_overall_for_model(df, meta=meta)
    
    # Print summary table
    print("\nDetailed summary by lang/domain:")
    print(df[['lang', 'domain', 'fertility', 'compression_chars', 'compression_bytes', 'reversible_ratio', 'unk_ratio', 'byte_fidelity', 'effective_bytes_per_token', 'ebpb', 'total_tokens', 'total_words', 'total_bytes', 'decoded_bytes_total', 'total_chars', 'total_docs', 'reversible_docs_count', 'unk_tokens_count', 'byte_edit_distance_total', 'byte_edit_denominator_total']].to_string(index=False))
    
    # Print language overalls
    if overalls.get("lang_overalls"):
        print("\nLanguage overalls:")
        
        lang_overalls_df = pd.DataFrame(overalls["lang_overalls"])
        parity_columns = [col for col in lang_overalls_df.columns if col.startswith('parity_')]
        metrics = ['fertility', 'compression_chars', 'compression_bytes', 'reversible_ratio', 'unk_ratio', 'byte_fidelity', 'effective_bytes_per_token', 'ebpb'] + parity_columns
        
        lang_overalls_df = lang_overalls_df[(['lang'] + metrics + ['total_tokens', 'total_words', 'total_chars', 'total_bytes', 'decoded_bytes_total', 'total_docs', 'reversible_docs_count', 'unk_tokens_count', 'byte_edit_distance_total', 'byte_edit_denominator_total', 'evaluation_date'])]

        # Add the multilingual overall fertility to the lang_overalls_df
        multilingual_row = {
            'lang': 'multilingual_summary',
            'fertility': lang_overalls_df['fertility'].mean(),
            'compression_chars': lang_overalls_df['compression_chars'].mean(),
            'compression_bytes': lang_overalls_df['compression_bytes'].mean(),
            'reversible_ratio': lang_overalls_df['reversible_ratio'].mean(),
            'unk_ratio': lang_overalls_df['unk_ratio'].mean(),
            'byte_fidelity': overalls['multilingual_overall'].get('byte_fidelity'),
            'effective_bytes_per_token': overalls['multilingual_overall'].get('effective_bytes_per_token'),
            'ebpb': overalls['multilingual_overall'].get('ebpb'),
            'total_tokens': lang_overalls_df['total_tokens'].sum(),
            'total_words': lang_overalls_df['total_words'].sum(),
            'total_chars': lang_overalls_df['total_chars'].sum(),
            'total_bytes': lang_overalls_df['total_bytes'].sum(),
            'decoded_bytes_total': lang_overalls_df['decoded_bytes_total'].sum(),
            'total_docs': lang_overalls_df['total_docs'].sum(),
            'reversible_docs_count': lang_overalls_df['reversible_docs_count'].sum(),
            'unk_tokens_count': lang_overalls_df['unk_tokens_count'].sum(),
            'byte_edit_distance_total': lang_overalls_df['byte_edit_distance_total'].sum(),
            'byte_edit_denominator_total': lang_overalls_df['byte_edit_denominator_total'].sum(),
            'evaluation_date': lang_overalls_df['evaluation_date'].max()
        }
        for col in parity_columns:
            multilingual_row[col] = lang_overalls_df[lang_overalls_df[col].notna()][col].mean()

        lang_overalls_df = pd.concat([
                lang_overalls_df, 
                pd.DataFrame([multilingual_row], columns=lang_overalls_df.columns)]
           ).reset_index(drop=True)

        print(lang_overalls_df.to_string(index=False))

        # Save lang_overalls_df to CSV if output_path is specified
        if args.output_path:
            try:
                output_target_path = Path(args.output_path)
                safe_name = generate_model_key(args.model_name, args.revision, args.subfolder, args.target_mode)

                csv_filename = f"lang_overalls_{safe_name}.csv"
                csv_output_path = output_target_path / csv_filename
                
                lang_overalls_df.to_csv(csv_output_path, index=False)
                print(f"Language overalls saved to {csv_output_path}")
            except Exception as e:
                print(f"Error saving language overalls to CSV: {e}")

    # Print multilingual overall if available
    if "multilingual_overall" in overalls:
        print("\nMultilingual overall:")
        print(f"Fertility: {overalls['multilingual_overall']['fertility']}")
        print(f"Compression (Chars): {overalls['multilingual_overall']['compression_chars']}")
        print(f"Compression (Bytes): {overalls['multilingual_overall']['compression_bytes']}")
        print(f"Reversible ratio: {overalls['multilingual_overall']['reversible_ratio']}")
        print(f"Unk ratio: {overalls['multilingual_overall']['unk_ratio']}")
        print(f"Vocab near duplicates: {overalls['multilingual_overall']['vocab_near_duplicates']*100:.2f}%")
        print(f"Total tokens: {overalls['multilingual_overall']['total_tokens']}")
        print(f"Total words: {overalls['multilingual_overall']['total_words']}")
        print(f"Total chars: {overalls['multilingual_overall']['total_chars']}")
        print(f"Total bytes: {overalls['multilingual_overall']['total_bytes']}")
        print(f"Decoded bytes total: {overalls['multilingual_overall'].get('decoded_bytes_total')}")
        print(f"Total docs: {overalls['multilingual_overall']['total_docs']}")
        print(f"Reversible Docs count: {overalls['multilingual_overall']['reversible_docs_count']}")
        print(f"Unk tokens count: {overalls['multilingual_overall']['unk_tokens_count']}")
        print(f"Byte edit distance total: {overalls['multilingual_overall'].get('byte_edit_distance_total')}")
        print(f"Byte edit denominator total: {overalls['multilingual_overall'].get('byte_edit_denominator_total')}")
        print(f"Last Eval Date: {overalls['multilingual_overall']['evaluation_date']}")