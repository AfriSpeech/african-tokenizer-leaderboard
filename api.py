import asyncio
import gzip as _gzip_mod
import traceback
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel


def _gzip_compress(data: bytes, level: int = 9) -> bytes:
    return _gzip_mod.compress(data, compresslevel=level)

from tokenizer_evaluate import (
    download_repos_once,
    load_dataset_meta,
    generate_model_key,
)
from util import ALL_COLUMNS, ensure_dataframe_has_columns
from results_aggregator import (
    get_overall_summary,
    calculate_summaries_vectorized,
    calculate_language_overalls_for_filters,
    load_all_model_results,
)
from submit import SubmissionWorker
import api_v2
from search_filter import search_filter
from env import APP_VERSION, LIMIT_MODELS_SHOW
from battle import (
    _tokenize_single_model as _battle_tokenize_single,
    tokenize_text as _battle_tokenize_text,
    create_model_choices,
    parse_model_selection,
)

# ---------------------------------------------------------------------------
# Column / display constants
# ---------------------------------------------------------------------------

GLOBAL_EXPECTED_COLUMNS = ALL_COLUMNS

DISPLAY_COLUMNS_MAP = {
    "model": "📛 Tokenizers",
    "ebpb": "📊 EBPB\n\t(bits/byte)",
    "fertility": "🪺 Fertility\n\t(tokens/word)",
    "compression_bytes": "🗃 Compression\n\t(bytes/token)",
    "parity_en": "⚖️ Parity\n\t(vs English)",
    # Experimental fidelity & frequency metrics — hidden by default
    "ebpb_full": "📊 EBPB Full\n\t(bits/byte, full vocab)",
    "effective_bytes_per_token": "📐 Eff. Bytes/Token\n\t(bytes/token)",
    "byte_fidelity": "🧬 Byte Fidelity\n\t(0–1 score)",
    "ebpc": "📈 EBPC\n\t(bits/char)",
    "effective_chars_per_token": "📏 Eff. Chars/Token\n\t(chars/token)",
    "char_fidelity": "🔤 Char Fidelity\n\t(0–1 score)",
    # Frequency / Zipf distribution metrics — hidden by default
    "freq_shannon_efficiency": "💡 Shannon Eff.\n\t(obs vocab)",
    "freq_renyi_efficiency": "💠 Rényi Eff.\n\t(obs vocab, α=3)",
    "freq_shannon_efficiency_full": "💡 Shannon Eff.\n\t(full vocab)",
    "freq_renyi_efficiency_full": "💠 Rényi Eff.\n\t(full vocab, α=3)",
    "freq_percentile_freq": "📊 Mid-Freq\n\tMass (P3–P83)",
    "freq_auc": "📉 Zipf AUC\n\t(log-log)",
    "freq_slope": "〽 Zipf Slope\n\t(exponent)",
    "freq_power_law": "🔁 Power-Law\n\tDev. (MAE)",
    "freq_entropy": "🌀 Shannon\n\tEntropy (bits)",
    "freq_renyi_entropy": "🌊 Rényi\n\tEntropy (bits)",
    "freq_hapax_count": "1️⃣ Hapax\n\tCount",
    "freq_hapax_ratio": "1️⃣ Hapax\n\tRatio (% types)",
    "freq_half_mass_count": "⚖️ Half-Mass\n\tTypes",
    "freq_gini": "📐 Token Gini\n\t(inequality)",
    #other metrics
    "reversible_ratio": "🔄 Reversible\n\t(% docs)",
    "unk_ratio": "❓ Unknown\n\t(% tokens)",
    #vocab metrics
    "total_tokens": "➕ Total Number\n\tof Tokens",
    "total_bytes": "💾 Total Bytes\n\t(UTF-8)",
    "total_chars": "🔤 Total Chars\n\t(Unicode)",
    "byte_edit_distance_total": "📏 Byte Edit\n\tDistance",
    "char_edit_distance_total": "📏 Char Edit\n\tDistance",
    "freq_cardinality": "🔢 Tokens Seen\n\t(unique count)", #exception should be close to vocab
    "freq_cardinality_ratio": "🔢 Tokens Seen\n\t(% vocab)",
    "freq_90pct_mass_count": "⚖️ 90%-Mass\n\tTypes",
    "freq_90pct_mass_ratio": "⚖️ 90%-Mass\n\t(% vocab)", #exception should be close to vocab
    "vocab_size": "📘 Vocab\n\tSize",
    "vocab_near_duplicates": "👥 Near\n\tDuplicates\n\t(% vocab)",
    "tokenizer_algorithm": "⚙️ Algorithm\n\t(Framework)",
    "vocab_hash_simple": "🔑 Tokenizer\n\tHash",
}

DEFAULT_VISIBLE_COLUMNS = [
    "model", "ebpb", "fertility", "compression_bytes", "parity_en",
    "char_fidelity", "freq_cardinality_ratio", "vocab_size",
    "vocab_near_duplicates", "tokenizer_algorithm", "vocab_hash_simple",
]

AVG_MULTILINGUAL_COLUMNS = {
    "fertility",
    "compression_bytes",
    "parity_en",
    "reversible_ratio",
    "unk_ratio",
    "byte_fidelity",
    "effective_bytes_per_token",
    "ebpb",
    "ebpb_full",
    "char_fidelity",
    "effective_chars_per_token",
    "ebpc",
    "freq_cardinality",
    "freq_cardinality_ratio",
    "freq_auc",
    "freq_slope",
    "freq_power_law",
    "freq_entropy",
    "freq_renyi_entropy",
    "freq_shannon_efficiency",
    "freq_renyi_efficiency",
    "freq_shannon_efficiency_full",
    "freq_renyi_efficiency_full",
    "freq_percentile_freq",
    "freq_hapax_count",
    "freq_hapax_ratio",
    "freq_half_mass_count",
    "freq_90pct_mass_count",
    "freq_90pct_mass_ratio",
    "freq_gini",
}

DEFAULT_SEARCH_QUERY = "@char_fidelity>0.9"

COLUMN_DESCRIPTIONS = {
    "model": "The tokenizer model name with a link to its Hugging Face page.",
    "fertility": "Fertility Score (Tokens/Word): Average number of tokens generated per word. Lower is better — fewer tokens means more efficient representation.",
    "compression_bytes": "Compression (Bytes/Token): Average number of UTF-8 bytes per token. Higher is better — more bytes per token means more compact representation.",
    "parity_en": "Parity vs English: Ratio of tokens for this language vs English on the same text. Closer to 1.0 is better. Above 1.0 means the tokenizer uses more tokens for this language than for English (less efficient, higher inference cost for non-English speakers). Below 1.0 means it is more efficient for this language than for English. Not applicable for English or Programming Languages (English is the reference).",
    "char_fidelity": "Char Fidelity: Corpus-level character reconstruction fidelity via normalized Levenshtein edit distance on the raw Unicode string (vs. byte_fidelity which operates on UTF-8 bytes). 1.0 = perfect reconstruction; lower values reveal character-level lossy decoding. More sensitive to multi-byte characters than byte_fidelity.",
    "reversible_ratio": "Reversible (% docs): Percentage of documents of the benchmark perfectly reconstructed after tokenize→decode. Higher is better — 100% indicates lossless encoding of all examples.",
    "unk_ratio": "Unknown (% tokens): Percentage of tokens of the benchmark identified as unknown. Lower is better — indicates better vocabulary coverage.",
    "total_tokens": "Total Number of Tokens: Total tokens generated across all selected datasets of the benchmark.",
    "total_bytes": "Total Bytes (UTF-8): Total number of UTF-8 bytes in the original corpus across all selected datasets.",
    "total_chars": "Total Chars (Unicode): Total number of Unicode characters in the original corpus across all selected datasets.",
    "byte_edit_distance_total": "Byte Edit Distance: Total Levenshtein edit distance between original and decoded text at the UTF-8 byte level, summed across the corpus. Used to derive byte_fidelity and EBPB. Lower is better — zero means perfect round-trip.",
    "char_edit_distance_total": "Char Edit Distance: Total Levenshtein edit distance between original and decoded text at the Unicode character level, summed across the corpus. Used to derive char_fidelity and EBPC. Lower is better.",
    "vocab_size": "Vocabulary Size: Number of unique tokens in the tokenizer's vocabulary. Larger vocabularies tend to have better metrics but higher embedding cost.",
    "vocab_near_duplicates": "Near Duplicates (% vocab): Percentage of near-duplicate tokens on the tokenizer's vocabulary (differing only in case, prefixes, or whitespace). Lower is better.",
    "tokenizer_algorithm": "Algorithm (Framework): The tokenization algorithm (BPE, WordPiece, Unigram, etc.) and implementation framework (hf-f (fast Hugging Face tokenizers), hf-s (slow Hugging Face transformers), sp (SentencePiece), tik (Tiktoken), fbpe (FastBPE)).",
    "vocab_hash_simple": "Tokenizer Hash (first 8 chars): Unique identifier computed from the tokenizer vocabulary and config. Helps identify tokenizers submitted under different models.",
    "byte_fidelity": "Byte Fidelity: Corpus-level byte reconstruction fidelity — measures how closely the reconstructed text with after tokenize→decode matches the original at the UTF-8 byte level using normalized Levenshtein edit distance. 1.0 = perfect reconstruction; lower values indicate character-level loss beyond unknown-token boundaries. Replaces the binary reversible_ratio with a continuous measure.",
    "effective_bytes_per_token": "Effective Bytes per Token: Compression rate adjusted for byte fidelity. Defined as compression_bytes × byte_fidelity. A tokenizer that drops characters scores lower than its raw bytes/token value. Higher is better.",
    "ebpb": "EBPB — Effective Bits Per Byte: Rate-distortion composite metric defined as (T × log₂(V_obs) + 8 × D_byte) / B, where T = total_tokens, V_obs = max(freq_cardinality, 2), D_byte = byte_edit_distance_total, B = total_bytes. Combines per-token indexing cost (using the observed vocabulary, not the nominal size) with a raw-byte fallback charge for any byte the tokenizer fails to reconstruct. Lower is better — fewer bits per original byte of corpus.",
    "ebpb_full": "EBPB Full — Effective Bits Per Byte (full vocabulary): Same rate-distortion formula as EBPB but uses the nominal vocabulary size (vocab_size) instead of observed cardinality. Defined as (T × log₂(V) + 8 × D_byte) / B. Penalises tokenizers with large declared vocabularies regardless of how many entries they actually use in practice. Lower is better.",
    "effective_chars_per_token": "Effective Chars per Token: Compression rate (chars/token) adjusted for character fidelity. Defined as compression_chars × char_fidelity. Higher is better.",
    "ebpc": "EBPC — Effective Bits Per Char: log2(vocab_size) / effective_chars_per_token. Character-space sibling of EBPB. Combines vocabulary expressiveness and fidelity-adjusted character compression. Lower is better.",
    "freq_cardinality": "Token Cardinality: Number of unique token IDs observed across the evaluation corpus. Captures how many distinct tokens the tokenizer actually uses (may be smaller than vocab_size). Higher values indicate broader vocabulary utilization.",
    "freq_cardinality_ratio": "Tokens Seen (% Vocab): Token cardinality divided by tokenizer vocabulary size. Shows what percentage of the full vocabulary appears in the evaluation corpus. Higher values indicate broader vocabulary utilization.",
    "freq_auc": "Zipf AUC: Area under the log-rank / log-frequency curve (Simpson's rule integration). A proxy for the total information in the rank-frequency distribution. Higher values indicate a richer, more spread-out token usage.",
    "freq_slope": "Zipf Slope: Slope of a linear fit on the log-rank / log-frequency curve restricted to the first ~403 ranks (log(rank) ≤ 6), approximating the Zipf exponent. More negative slopes indicate steeper frequency drop-off (fewer dominant tokens). Based on Lotz et al. (2025, arXiv:2506.03101).",
    "freq_power_law": "Power-Law Deviation: Mean absolute error of the Zipf linear fit on the log-rank / log-frequency curve. Lower values indicate the distribution follows a purer power law. Based on Lotz et al. (2025, arXiv:2506.03101).",
    "freq_entropy": "Shannon Entropy: Shannon entropy (in bits) of the token frequency distribution over the corpus. Higher values indicate more uniform token usage and better vocabulary utilization.",
    "freq_renyi_entropy": "Rényi Entropy (α=3): Rényi entropy of order 3 (in bits) of the token frequency distribution. More sensitive to high-frequency tokens than Shannon entropy. Identified by Zouhar et al. (ACL 2023) as strongly predictive of downstream MT performance. Higher is better.",
    "freq_shannon_efficiency": "Shannon Efficiency (obs. vocab): Shannon entropy normalized by log2(observed unique tokens). Ranges 0–1; closer to 1 means the tokenizer uses its observed types more uniformly. Higher is better.",
    "freq_renyi_efficiency": "Rényi Efficiency (obs. vocab, α=3): Rényi entropy (α=3) normalized by log2(observed unique tokens). Identified by Zouhar et al. (ACL 2023) as the strongest single predictor of downstream MT performance. Higher is better.",
    "freq_shannon_efficiency_full": "Shannon Efficiency (full vocab): Shannon entropy normalized by log2(tokenizer.vocab_size). Accounts for unseen tokens; lower than the observed-vocab version when the tokenizer has many unused entries. Higher is better.",
    "freq_renyi_efficiency_full": "Rényi Efficiency (full vocab, α=3): Rényi entropy (α=3) normalized by log2(tokenizer.vocab_size). Full-vocabulary sibling of freq_renyi_efficiency. Higher is better.",
    "freq_percentile_freq": "Mid-Frequency Mass (P3–P83): Sum of token probabilities from the 3rd to the 83rd percentile of the descending frequency ranking. Captures how much of the corpus is covered by mid-frequency tokens. Identified by Zouhar et al. (ACL 2023) as the second-best predictor of MT performance after Rényi efficiency. Higher is better.",
    "freq_hapax_count": "Hapax Count: Number of unique token types observed exactly once in the corpus. High hapax counts may indicate tokenizer fragmentation or domain mismatch.",
    "freq_hapax_ratio": "Hapax Ratio: Fraction of unique observed token types that appear exactly once. Lower values indicate the tokenizer uses its vocabulary more consistently across the corpus.",
    "freq_half_mass_count": "Half-Mass Types: Number of unique token types needed to cover 50% of all token occurrences (sorted by descending frequency). Smaller values indicate a more head-concentrated, less uniform distribution.",
    "freq_90pct_mass_count": "90%-Mass Types: Number of unique token types needed to cover 90% of all token occurrences. Higher values indicate broader, more uniform vocabulary usage.",
    "freq_90pct_mass_ratio": "90%-Mass Types (% Vocab): Number of unique token types needed to cover 90% of all token occurrences divided by tokenizer vocabulary size. Higher values indicate broader corpus coverage across the vocabulary.",
    "freq_gini": "Token Gini: Gini inequality coefficient of the token frequency distribution. 0 = perfectly uniform use of all observed types, 1 = a single token dominates all occurrences. Lower is better (more diverse, equitable token usage).",
}

_DEFAULT_VISIBLE_SET = set(DEFAULT_VISIBLE_COLUMNS)

def get_column_descriptions() -> dict:
    """Return COLUMN_DESCRIPTIONS with '[Experimental]' injected after the title
    for every key that is not in DEFAULT_VISIBLE_COLUMNS."""
    result = {}
    for key, desc in COLUMN_DESCRIPTIONS.items():
        if key not in _DEFAULT_VISIBLE_SET:
            colon = desc.find(':')
            if colon != -1:
                desc = desc[:colon] + ' [Experimental]' + desc[colon:]
        result[key] = desc
    return result

FILTER_FIELDS = [
    {"field": "fertility", "description": "Tokens per word"},
    {"field": "compression_bytes", "description": "Bytes per token"},
    {"field": "parity_en", "description": "Token ratio vs English"},
    {"field": "reversible_ratio", "description": "Reversible docs ratio (0–1 or 0–100%)"},
    {"field": "unk_ratio", "description": "Unknown tokens ratio (0–1 or 0–100%)"},
    {"field": "vocab_size", "description": "Vocabulary size"},
    {"field": "vocab_near_duplicates", "description": "Near duplicate % in vocabulary"},
    {"field": "total_tokens", "description": "Total number of tokens"},
    {"field": "total_bytes", "description": "Total UTF-8 bytes in the corpus"},
    {"field": "total_chars", "description": "Total Unicode characters in the corpus"},
    {"field": "byte_edit_distance_total", "description": "Total byte-level Levenshtein edit distance (lower is better)"},
    {"field": "char_edit_distance_total", "description": "Total char-level Levenshtein edit distance (lower is better)"},
    {"field": "byte_fidelity", "description": "Byte fidelity score (0–1)"},
    {"field": "effective_bytes_per_token", "description": "Fidelity-adjusted bytes per token"},
    {"field": "ebpb", "description": "Effective bits per byte — observed-cardinality indexing plus raw-byte fallback cost (lower is better)"},
    {"field": "ebpb_full", "description": "Effective bits per byte using full declared vocabulary size (lower is better)"},
    {"field": "char_fidelity", "description": "Char fidelity score (0–1)"},
    {"field": "effective_chars_per_token", "description": "Fidelity-adjusted chars per token"},
    {"field": "ebpc", "description": "Effective bits per char (lower is better)"},
    {"field": "freq_cardinality", "description": "Unique token types observed"},
    {"field": "freq_cardinality_ratio", "description": "Unique token types observed as fraction of vocab size (0–1)"},
    {"field": "freq_auc", "description": "Area under log-rank/log-freq curve (Zipf)"},
    {"field": "freq_slope", "description": "Zipf slope (log-rank/log-freq linear fit)"},
    {"field": "freq_power_law", "description": "Power-law deviation from Zipf fit (MAE)"},
    {"field": "freq_entropy", "description": "Shannon entropy of token frequencies (bits)"},
    {"field": "freq_renyi_entropy", "description": "Rényi entropy α=3 of token frequencies (bits)"},
    {"field": "freq_shannon_efficiency", "description": "Shannon efficiency (observed vocab, 0–1)"},
    {"field": "freq_renyi_efficiency", "description": "Rényi efficiency α=3 (observed vocab, 0–1) — best MT predictor"},
    {"field": "freq_shannon_efficiency_full", "description": "Shannon efficiency (full vocab, 0–1)"},
    {"field": "freq_renyi_efficiency_full", "description": "Rényi efficiency α=3 (full vocab, 0–1)"},
    {"field": "freq_percentile_freq", "description": "Mid-frequency token mass (P3–P83, 0–1)"},
    {"field": "freq_hapax_count", "description": "Number of once-only token types"},
    {"field": "freq_hapax_ratio", "description": "Fraction of once-only token types (0–1)"},
    {"field": "freq_half_mass_count", "description": "Types needed to cover 50% of corpus"},
    {"field": "freq_90pct_mass_count", "description": "Types needed to cover 90% of corpus"},
    {"field": "freq_90pct_mass_ratio", "description": "Types needed to cover 90% of corpus as fraction of vocab size (0–1)"},
    {"field": "freq_gini", "description": "Gini inequality of token frequencies (0–1, lower=more uniform)"},
]

# ---------------------------------------------------------------------------
# Data-processing helpers
# ---------------------------------------------------------------------------

def get_dataframe_datatypes(column_map):
    datatypes = []
    percentage_columns = {
        "reversible_ratio", "unk_ratio", "vocab_near_duplicates", "byte_fidelity",
        "char_fidelity",
        "freq_hapax_ratio", "freq_gini",
        "freq_shannon_efficiency", "freq_renyi_efficiency",
        "freq_shannon_efficiency_full", "freq_renyi_efficiency_full",
        "freq_percentile_freq",
        "freq_cardinality_ratio",
        "freq_90pct_mass_ratio",
    }
    numeric_columns = {
        "fertility",
        "compression_chars",
        "compression_bytes",
        "parity_en",
        "total_tokens",
        "total_bytes",
        "total_chars",
        "byte_edit_distance_total",
        "char_edit_distance_total",
        "vocab_size",
        "effective_bytes_per_token",
        "ebpb",
        "ebpb_full",
        "effective_chars_per_token",
        "ebpc",
        "freq_cardinality",
        "freq_auc",
        "freq_slope",
        "freq_power_law",
        "freq_entropy",
        "freq_renyi_entropy",
        "freq_hapax_count",
        "freq_half_mass_count",
        "freq_90pct_mass_count",
    }
    for original_col in column_map.keys():
        if original_col == "model":
            datatypes.append("str")
        elif original_col in percentage_columns or original_col in numeric_columns:
            datatypes.append("number")
        else:
            datatypes.append("str")
    return datatypes


def filter_duplicates(df, hide=True, sort_col="fertility", sort_asc=True):
    from tokenizer_evaluate import get_tokenizer_hash_field

    tokenizer_hash_field = get_tokenizer_hash_field()
    if (
        df.empty
        or tokenizer_hash_field not in df.columns
        or "evaluation_date" not in df.columns
    ):
        return df.copy()

    df_copy = df.copy()
    df_copy["is_main"] = df_copy["revision"].fillna("main") == "main"
    df_copy["is_target"] = df_copy["_target_mode"].fillna(False)
    df_copy["is_subfolder"] = df_copy["subfolder"].fillna("") != ""
    df_copy["likes"] = df_copy["likes"].fillna(0).astype(int)
    df_copy["author_is_verified"] = df_copy["author_is_verified"].fillna(False).astype(bool) if "author_is_verified" in df_copy.columns else False
    df_copy["author_followers"] = df_copy["author_followers"].fillna(0).astype(int) if "author_followers" in df_copy.columns else 0
    df_copy["evaluation_date"] = pd.to_datetime(
        df_copy["evaluation_date"], errors="coerce"
    )
    df_copy["created_at"] = pd.to_datetime(df_copy["created_at"], errors="coerce")

    filtered_df = df_copy.sort_values(
        [
            sort_col,
            tokenizer_hash_field,
            "author_is_verified",
            "author_followers",
            "likes",
            "created_at",
            "is_main",
            "is_target",
            "is_subfolder",
            "model_key",
        ],
        ascending=[sort_asc, True, False, False, False, True, False, True, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    if not hide:
        return filtered_df

    filtered_df = filtered_df.drop_duplicates(
        subset=[tokenizer_hash_field], keep="first"
    ).reset_index(drop=True)
    return filtered_df


def _select_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select columns for API response. No value transformations — raw values pass through."""
    if df is None or df.empty:
        return pd.DataFrame()

    from tokenizer_evaluate import get_tokenizer_hash_field
    tokenizer_hash_field = get_tokenizer_hash_field()

    desired = [
        # Display columns
        "model", "revision", "subfolder", "_target_mode",
        "fertility", "compression_bytes", "parity_en",
        "reversible_ratio", "unk_ratio", "total_tokens", "vocab_size",
        "vocab_near_duplicates", "tokenizer_algorithm", "tokenizer_implementation",
        # Experimental fidelity metrics
        "total_bytes", "total_chars", "byte_edit_distance_total", "char_edit_distance_total",
        "byte_fidelity", "effective_bytes_per_token", "ebpb", "ebpb_full",
        "char_fidelity", "effective_chars_per_token", "ebpc",
        # Experimental frequency / Zipf metrics
        "freq_cardinality", "freq_cardinality_ratio", "freq_auc", "freq_slope", "freq_power_law",
        "freq_entropy", "freq_renyi_entropy",
        "freq_shannon_efficiency", "freq_renyi_efficiency",
        "freq_shannon_efficiency_full", "freq_renyi_efficiency_full",
        "freq_percentile_freq",
        "freq_hapax_count", "freq_hapax_ratio", "freq_half_mass_count",
        "freq_90pct_mass_count", "freq_90pct_mass_ratio", "freq_gini",
        # Extra columns for frontend dedup + search
        "vocab_hash", "likes", "created_at", "model_key",
        "author_is_verified", "author_followers",
    ]
    available = [c for c in desired if c in df.columns]
    # Always include the primary hash field (may be tokenizer_config_hash or vocab_hash)
    if tokenizer_hash_field not in available and tokenizer_hash_field in df.columns:
        available.append(tokenizer_hash_field)

    result = df[available].copy()

    # vocab_hash_simple: 8-char display identifier from the primary hash field
    if tokenizer_hash_field in result.columns:
        result["vocab_hash_simple"] = result[tokenizer_hash_field].str[:8]
        # Keep the full hash too (renamed to tokenizer_config_hash for the frontend)
        if tokenizer_hash_field != "tokenizer_config_hash":
            result = result.rename(columns={tokenizer_hash_field: "tokenizer_config_hash"})

    if "_target_mode" in result.columns:
        result = result.rename(columns={"_target_mode": "target_mode"})

    # Nullify parity_en when value is 0 or NaN — these are "not applicable" cases
    # (e.g. English tab is the reference language, programming languages have no parallel corpus)
    if "parity_en" in result.columns:
        result["parity_en"] = result["parity_en"].where(
            result["parity_en"].notna() & (result["parity_en"] != 0), other=None
        )

    return result


# ---------------------------------------------------------------------------
# FilterState / ComputeResult dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FilterState:
    selected_langs: List[str] = field(default_factory=list)
    selected_domains: Dict[str, List[str]] = field(default_factory=dict)
    selected_multilingual_domains: List[str] = field(default_factory=list)

    def __eq__(self, other):
        if other is None or not isinstance(other, FilterState):
            return False
        if set(self.selected_langs) != set(other.selected_langs):
            return False
        if set(self.selected_multilingual_domains) != set(
            other.selected_multilingual_domains
        ):
            return False
        for lang_code, domains in self.selected_domains.items():
            if lang_code not in other.selected_domains:
                return False
            elif set(domains) != set(other.selected_domains[lang_code]):
                return False
        return True


@dataclass
class ComputeResult:
    """Full leaderboard computation output — no limit applied.

    The limit is applied later at response-serialization time in
    _build_leaderboard_response, keeping computation and payload size decoupled.
    """
    multilingual_df: pd.DataFrame
    multilingual_total_count: int
    multilingual_filtered_count: int
    lang_codes: List[str]
    lang_dfs: List[pd.DataFrame]
    lang_total_counts: List[int]
    lang_filtered_counts: List[int]


# ---------------------------------------------------------------------------
# GlobalAppStateCache — read-only data + pre-computed default result
# ---------------------------------------------------------------------------

class GlobalAppStateCache:
    def __init__(self):
        self.is_initialized = False
        # Read-only source data (loaded once from disk, never mutated during requests)
        self.raw_results_df: pd.DataFrame = pd.DataFrame()
        self.df_lang_overalls: pd.DataFrame = pd.DataFrame()
        self.df_multilingual_raw: pd.DataFrame = pd.DataFrame()
        self.available_models_with_metadata: list = []
        # Pre-computed result for the default filter state (serves new connections fast)
        self.default_filter_state: Optional[FilterState] = None
        self.default_compute_result: Optional[ComputeResult] = None
        # Cached metadata response (static after startup)
        self.metadata_cache: Optional[dict] = None
        # Pre-serialized JSON bytes for default responses (zero-compute hot path)
        self._json_metadata: Optional[bytes] = None
        self._json_leaderboard_init: Optional[bytes] = None
        self._json_lang_default: Dict[str, bytes] = {}
        self._json_multilingual_default: Optional[bytes] = None
        # Pre-compressed gzip versions of the above
        self._gz_metadata: Optional[bytes] = None
        self._gz_leaderboard_init: Optional[bytes] = None
        self._gz_lang_default: Dict[str, bytes] = {}
        self._gz_multilingual_default: Optional[bytes] = None
        # Default domain/lang sets for cache-hit detection on sub-endpoints
        self._default_lang_domains: Dict[str, set] = {}
        self._default_multilingual_langs: set = set()
        self._default_multilingual_domains: set = set()
        # Data version for ETag-based cache invalidation
        self._data_version: str = "0"

    def initialize(
        self, benchmark_dir, results_dir, expected_columns_list, force_reload=False
    ):
        if self.is_initialized and not force_reload:
            print("GlobalAppStateCache: Already initialized, skipping...")
            return
        print("GlobalAppStateCache: Loading heavy data...")
        self.raw_results_df = ensure_dataframe_has_columns(
            load_all_model_results(results_dir), expected_columns_list
        )
        results_summary = get_overall_summary(
            benchmark_dir, results_dir, self.raw_results_df
        )
        results_summary["lang_overalls"] = ensure_dataframe_has_columns(
            results_summary["lang_overalls"], expected_columns_list
        )
        results_summary["multilingual_overall"] = ensure_dataframe_has_columns(
            results_summary["multilingual_overall"], expected_columns_list
        )
        self.df_lang_overalls = results_summary.get("lang_overalls")
        if self.df_lang_overalls.empty:
            self.df_lang_overalls = ensure_dataframe_has_columns(
                pd.DataFrame(columns=expected_columns_list), expected_columns_list
            )
        self.df_multilingual_raw = results_summary.get("multilingual_overall")
        if self.df_multilingual_raw.empty:
            self.df_multilingual_raw = ensure_dataframe_has_columns(
                pd.DataFrame(columns=expected_columns_list), expected_columns_list
            )
        if not self.df_multilingual_raw.empty:
            unique_models = self.df_multilingual_raw[
                ["model", "revision", "subfolder", "_target_mode"]
            ].drop_duplicates()
            models_data = []
            for _, row in unique_models.iterrows():
                model_info = {
                    "model": row["model"],
                    "revision": row["revision"]
                    if pd.notna(row["revision"])
                    else "main",
                    "subfolder": row["subfolder"]
                    if pd.notna(row["subfolder"])
                    else None,
                    "_target_mode": row["_target_mode"]
                    if pd.notna(row["_target_mode"])
                    else False,
                }
                models_data.append(model_info)
            models_data.sort(key=lambda x: x["model"])
            self.available_models_with_metadata = models_data
        else:
            self.available_models_with_metadata = []
        self.is_initialized = True
        print(
            f"GlobalAppStateCache: Initialization complete. Found {len(self.available_models_with_metadata)} model variants."
        )

    def refresh(self, benchmark_dir, results_dir, expected_columns_list):
        print("GlobalAppStateCache: Force refreshing cache...")
        self.initialize(
            benchmark_dir, results_dir, expected_columns_list, force_reload=True
        )
        # Invalidate all pre-computed defaults so they get rebuilt after refresh
        self.default_filter_state = None
        self.default_compute_result = None
        self.metadata_cache = None
        self._json_metadata = None
        self._json_leaderboard_init = None
        self._json_lang_default = {}
        self._json_multilingual_default = None
        self._gz_metadata = None
        self._gz_leaderboard_init = None
        self._gz_lang_default = {}
        self._gz_multilingual_default = None
        self._default_lang_domains = {}
        self._default_multilingual_langs = set()
        self._default_multilingual_domains = set()
        self._data_version = "0"


# ---------------------------------------------------------------------------
# Module-level singletons (populated in lifespan, read-only after startup)
# ---------------------------------------------------------------------------

_GLOBAL_CACHE = GlobalAppStateCache()

benchmark_dir: Path = None  # type: ignore[assignment]
results_dir: Path = None  # type: ignore[assignment]
benchmark_file_hashes: dict = {}
tokenizer_hash_cache: dict = {}
dataset_meta: dict = {}
language_tabs: list = []
code_domains: set = set()
submission_worker: SubmissionWorker = None  # type: ignore[assignment]


def get_available_domains_for_languages(selected_lang_codes):
    if not selected_lang_codes:
        return []
    all_domains = set()
    for lang_tab in language_tabs:
        if lang_tab["code"] in selected_lang_codes and lang_tab["code"] != "code":
            for pretty_name, domain_name in lang_tab["domains"]:
                all_domains.add((pretty_name, domain_name))
    all_domains = sorted(list(all_domains), key=lambda x: x[1])
    if "code" in selected_lang_codes:
        all_domains.insert(0, ("Programming Languages", "code"))
    return all_domains


def get_all_available_domains():
    all_domains = set()
    for lang_tab in language_tabs:
        if lang_tab["code"] != "code":
            for pretty_name, domain_name in lang_tab["domains"]:
                all_domains.add((pretty_name, domain_name))
    all_domains = sorted(list(all_domains), key=lambda x: x[1])
    all_domains.insert(0, ("Programming Languages", "code"))
    return all_domains


def initialize_global_cache():
    _GLOBAL_CACHE.initialize(benchmark_dir, results_dir, GLOBAL_EXPECTED_COLUMNS)


def refresh_global_cache():
    _GLOBAL_CACHE.refresh(benchmark_dir, results_dir, GLOBAL_EXPECTED_COLUMNS)


def _make_default_filter_state() -> FilterState:
    return FilterState(
        selected_langs=[lang["code"] for lang in language_tabs],
        selected_domains={
            lang["code"]: [name for pretty, name in lang["domains"]]
            for lang in language_tabs
        },
        selected_multilingual_domains=[
            name for pretty, name in get_all_available_domains()
        ],
    )


# ---------------------------------------------------------------------------
# Pure filter computation — no global state mutations, safe for concurrent calls
# ---------------------------------------------------------------------------

def _compute_multilingual(
    selected_langs: List[str],
    selected_multilingual_domains: List[str],
) -> tuple:
    """Compute the multilingual aggregate DataFrame.

    Returns (df, total_count, filtered_count).
    Reads only from _GLOBAL_CACHE (read-only after startup).
    """
    all_available_langs = [lang["code"] for lang in language_tabs]
    all_available_domains = [name for pretty, name in get_all_available_domains()]
    all_langs_selected = set(selected_langs) == set(all_available_langs)
    all_domains_selected = set(selected_multilingual_domains) == set(all_available_domains)

    if all_langs_selected and all_domains_selected:
        multilingual_df_pre = _GLOBAL_CACHE.df_multilingual_raw.copy()
    elif all_domains_selected and not all_langs_selected:
        filtered_lang_overalls = _GLOBAL_CACHE.df_lang_overalls[
            (_GLOBAL_CACHE.df_lang_overalls["lang"].isin(selected_langs))
            & (_GLOBAL_CACHE.df_lang_overalls["domain"] == "lang_overall")
        ].copy()
        if not filtered_lang_overalls.empty:
            selected_languages_set = set(selected_langs)
            model_lang_sets = filtered_lang_overalls.groupby("model_key")["lang"].apply(set)
            models_with_all_langs = model_lang_sets[
                model_lang_sets.apply(lambda langs: selected_languages_set.issubset(langs))
            ].index
            complete_lang_overalls = filtered_lang_overalls[
                filtered_lang_overalls["model_key"].isin(models_with_all_langs)
            ].copy()
            if not complete_lang_overalls.empty:
                multilingual_df_pre = calculate_summaries_vectorized(
                    complete_lang_overalls, is_multilingual=True
                )
            else:
                multilingual_df_pre = pd.DataFrame(columns=GLOBAL_EXPECTED_COLUMNS)
        else:
            multilingual_df_pre = pd.DataFrame(columns=GLOBAL_EXPECTED_COLUMNS)
    else:
        _selected_domains = list(selected_multilingual_domains)
        if "code" in _selected_domains:
            _selected_domains.remove("code")
            for domain in code_domains:
                _selected_domains.append(domain)
        lang_overalls_filtered = calculate_language_overalls_for_filters(
            _GLOBAL_CACHE.raw_results_df,
            _selected_domains,
            selected_langs,
            GLOBAL_EXPECTED_COLUMNS,
        )
        if not lang_overalls_filtered.empty:
            effective_selected_langs = set(lang_overalls_filtered["lang"].unique())
            model_lang_sets = _GLOBAL_CACHE.df_lang_overalls.groupby("model_key")["lang"].apply(set)
            models_with_all_langs = model_lang_sets[
                model_lang_sets.apply(lambda langs: effective_selected_langs.issubset(langs))
            ].index
            complete_lang_overalls = lang_overalls_filtered[
                lang_overalls_filtered["model_key"].isin(models_with_all_langs)
            ].copy()
            if not complete_lang_overalls.empty:
                multilingual_df_pre = calculate_summaries_vectorized(
                    complete_lang_overalls, is_multilingual=True
                )
            else:
                multilingual_df_pre = pd.DataFrame(columns=GLOBAL_EXPECTED_COLUMNS)
        else:
            multilingual_df_pre = pd.DataFrame(columns=GLOBAL_EXPECTED_COLUMNS)

    total_count = len(multilingual_df_pre)
    df = _select_display_columns(multilingual_df_pre)
    filtered_count = len(df)
    return df, total_count, filtered_count


def _compute_single_lang(lang_code: str, selected_domains_for_lang: List[str]) -> tuple:
    """Compute the DataFrame for a single language tab.

    Returns (df, total_count, filtered_count).
    Reads only from _GLOBAL_CACHE (read-only after startup).
    """
    available_domains_for_lang = []
    for lang_tab in language_tabs:
        if lang_tab["code"] == lang_code:
            available_domains_for_lang = [name for pretty, name in lang_tab["domains"]]
            break

    all_domains_selected_for_lang = set(selected_domains_for_lang) == set(available_domains_for_lang)

    if all_domains_selected_for_lang:
        lang_overalls_df = _GLOBAL_CACHE.df_lang_overalls[
            (_GLOBAL_CACHE.df_lang_overalls["lang"] == lang_code)
            & (_GLOBAL_CACHE.df_lang_overalls["domain"] == "lang_overall")
        ].copy()
    else:
        lang_overalls_df = pd.DataFrame(columns=GLOBAL_EXPECTED_COLUMNS)
        if selected_domains_for_lang:
            filtered_raw_df = _GLOBAL_CACHE.raw_results_df[
                (_GLOBAL_CACHE.raw_results_df["lang"] == lang_code)
                & (_GLOBAL_CACHE.raw_results_df["domain"].isin(selected_domains_for_lang))
                & (_GLOBAL_CACHE.raw_results_df["domain"] != "lang_overall")
                & (_GLOBAL_CACHE.raw_results_df["domain"] != "multilingual_overall")
                & (_GLOBAL_CACHE.raw_results_df["evaluation_type"] == "standard")
            ].copy()
            if not filtered_raw_df.empty:
                lang_model_df = _GLOBAL_CACHE.df_lang_overalls[
                    _GLOBAL_CACHE.df_lang_overalls["lang"] == lang_code
                ].copy()
                parity_cols = [col for col in lang_model_df.columns if col.startswith("parity_")]
                parity_lookup_df = None
                if parity_cols:
                    lookup_cols = ["model_key", "lang"] + parity_cols
                    parity_lookup_df = lang_model_df[lookup_cols].copy()
                lang_overalls_df = calculate_summaries_vectorized(
                    filtered_raw_df,
                    is_multilingual=False,
                    lookup_df=parity_lookup_df,
                )

    total_count = len(lang_overalls_df)
    df = _select_display_columns(lang_overalls_df)
    filtered_count = len(df)
    return df, total_count, filtered_count


def compute_leaderboard(filter_state: FilterState) -> ComputeResult:
    """Compute leaderboard data for a given FilterState.

    Delegates to _compute_multilingual and _compute_single_lang.
    Returns full DataFrames with no row limit applied.
    Does not mutate any global state.
    """
    multilingual_df, multilingual_total_count, multilingual_filtered_count = _compute_multilingual(
        filter_state.selected_langs,
        filter_state.selected_multilingual_domains,
    )

    lang_codes = []
    lang_dfs = []
    lang_total_counts = []
    lang_filtered_counts = []

    for lang_code, selected_domains_for_lang in filter_state.selected_domains.items():
        df, total_count, filtered_count = _compute_single_lang(lang_code, selected_domains_for_lang)
        lang_codes.append(lang_code)
        lang_dfs.append(df)
        lang_total_counts.append(total_count)
        lang_filtered_counts.append(filtered_count)

    return ComputeResult(
        multilingual_df=multilingual_df,
        multilingual_total_count=multilingual_total_count,
        multilingual_filtered_count=multilingual_filtered_count,
        lang_codes=lang_codes,
        lang_dfs=lang_dfs,
        lang_total_counts=lang_total_counts,
        lang_filtered_counts=lang_filtered_counts,
    )


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class FilterRequest(BaseModel):
    selected_langs: List[str] = []
    selected_domains: Dict[str, List[str]] = {}
    selected_multilingual_domains: List[str] = []
    limit: Optional[int] = LIMIT_MODELS_SHOW  # None = no limit


class LangFilterRequest(BaseModel):
    selected_domains: List[str] = []


class MultilingualFilterRequest(BaseModel):
    selected_langs: List[str] = []
    selected_multilingual_domains: List[str] = []


class SubmitRequest(BaseModel):
    model_name: str
    revision: str = "main"
    subfolder: Optional[str] = None
    target_mode: bool = False
    selected_langs: Optional[List[str]] = None


class BattleRequest(BaseModel):
    text: str
    model_selection_1: str
    model_selection_2: str
    better_tokenization: bool = False


class ValidateSearchRequest(BaseModel):
    query: str


class DomainsForLanguagesRequest(BaseModel):
    selected_lang_codes: List[str]


# ---------------------------------------------------------------------------
# Helpers: DataFrame -> JSON-safe records
# ---------------------------------------------------------------------------

def _df_to_records(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", default_handler=str))


def _build_leaderboard_response(
    result: ComputeResult,
    filter_state: FilterState,
    limit: Optional[int],
) -> dict:
    # limit is a hint to cap payload size; client-side filtering (search/dedup) is not done here
    m_df = result.multilingual_df.head(limit) if limit is not None else result.multilingual_df
    m_count_info = {
        "total": result.multilingual_total_count,
        "domain_filtered": result.multilingual_filtered_count,
    }

    lang_dfs_dict = {}
    lang_count_infos = {}
    for i, lang_code in enumerate(result.lang_codes):
        lang_df = result.lang_dfs[i]
        lang_df_display = lang_df.head(limit) if limit is not None else lang_df
        lang_count_infos[lang_code] = {
            "total": result.lang_total_counts[i],
            "domain_filtered": result.lang_filtered_counts[i],
        }
        lang_dfs_dict[lang_code] = _df_to_records(lang_df_display)

    models_with_metadata = _GLOBAL_CACHE.available_models_with_metadata or []
    model_choices = create_model_choices(models_with_metadata)

    return {
        "multilingual_df": _df_to_records(m_df),
        "multilingual_counts": m_count_info,
        "lang_dfs": lang_dfs_dict,
        "lang_count_infos": lang_count_infos,
        "model_choices": [{"display": d, "value": v} for d, v in model_choices],
        "filter_state": {
            "selected_langs": filter_state.selected_langs,
            "selected_domains": filter_state.selected_domains,
            "selected_multilingual_domains": filter_state.selected_multilingual_domains,
        },
    }


# ---------------------------------------------------------------------------
# Pre-serialization helper — called at startup and after refresh
# ---------------------------------------------------------------------------

def _preserialize_defaults(
    default_filter_state: FilterState,
    default_compute_result: ComputeResult,
) -> None:
    """Serialize all default (no-filter) responses to JSON bytes + gzip, and
    store them on _GLOBAL_CACHE so endpoints can return them with zero computation."""
    print("API: pre-serializing default responses...")

    # --- data version (from max evaluation_date) ---
    if "evaluation_date" in _GLOBAL_CACHE.raw_results_df.columns:
        max_eval = pd.to_datetime(
            _GLOBAL_CACHE.raw_results_df["evaluation_date"], errors="coerce"
        ).max()
        _GLOBAL_CACHE._data_version = str(max_eval) if pd.notna(max_eval) else "0"
    else:
        _GLOBAL_CACHE._data_version = "0"
    print(f"API: data_version = {_GLOBAL_CACHE._data_version}")

    # --- metadata ---
    metadata_dict = {
        "data_version": _GLOBAL_CACHE._data_version,
        "language_tabs": language_tabs,
        "all_domains": get_all_available_domains(),
        "display_columns_map": DISPLAY_COLUMNS_MAP,
        "default_visible_columns": DEFAULT_VISIBLE_COLUMNS,
        "avg_multilingual_columns": list(AVG_MULTILINGUAL_COLUMNS),
        "default_search_query": DEFAULT_SEARCH_QUERY,
        "filter_help_text": search_filter.get_filter_help(),
        "dataframe_datatypes": get_dataframe_datatypes(DISPLAY_COLUMNS_MAP),
        "limit_models_show": LIMIT_MODELS_SHOW,
        "column_descriptions": get_column_descriptions(),
        "filter_fields": FILTER_FIELDS,
    }
    _GLOBAL_CACHE.metadata_cache = metadata_dict
    _GLOBAL_CACHE._json_metadata = json.dumps(metadata_dict, default=str, ensure_ascii=False).encode("utf-8")
    _GLOBAL_CACHE._gz_metadata = _gzip_compress(_GLOBAL_CACHE._json_metadata)

    # --- full leaderboard (default filter) ---
    # Still needed for the POST /api/leaderboard default-filter fast path (uses pre-computed DataFrames).
    # No bytes serialized here — the endpoint calls _build_leaderboard_response directly.

    # --- per-language responses ---
    lang_default_domains: Dict[str, set] = {}
    for i, lang_code in enumerate(default_compute_result.lang_codes):
        lang_df = default_compute_result.lang_dfs[i]
        lang_df_display = lang_df.head(LIMIT_MODELS_SHOW) if LIMIT_MODELS_SHOW is not None else lang_df
        lang_dict = {
            "lang_df": _df_to_records(lang_df_display),
            "lang_counts": {
                "total": default_compute_result.lang_total_counts[i],
                "domain_filtered": default_compute_result.lang_filtered_counts[i],
            },
        }
        _GLOBAL_CACHE._json_lang_default[lang_code] = json.dumps(lang_dict, default=str, ensure_ascii=False).encode("utf-8")
        _GLOBAL_CACHE._gz_lang_default[lang_code] = _gzip_compress(_GLOBAL_CACHE._json_lang_default[lang_code])
        # Record the default domain set for this language for cache-hit detection
        for lang_tab in language_tabs:
            if lang_tab["code"] == lang_code:
                lang_default_domains[lang_code] = set(name for _, name in lang_tab["domains"])
                break
    _GLOBAL_CACHE._default_lang_domains = lang_default_domains

    # --- multilingual response ---
    m_df = default_compute_result.multilingual_df
    m_df_display = m_df.head(LIMIT_MODELS_SHOW) if LIMIT_MODELS_SHOW is not None else m_df
    multilingual_dict = {
        "multilingual_df": _df_to_records(m_df_display),
        "multilingual_counts": {
            "total": default_compute_result.multilingual_total_count,
            "domain_filtered": default_compute_result.multilingual_filtered_count,
        },
    }
    _GLOBAL_CACHE._json_multilingual_default = json.dumps(multilingual_dict, default=str, ensure_ascii=False).encode("utf-8")
    _GLOBAL_CACHE._gz_multilingual_default = _gzip_compress(_GLOBAL_CACHE._json_multilingual_default)
    _GLOBAL_CACHE._default_multilingual_langs = set(default_filter_state.selected_langs)
    _GLOBAL_CACHE._default_multilingual_domains = set(default_filter_state.selected_multilingual_domains)

    # --- leaderboard init response (multilingual only, no per-lang data) ---
    # Reuses the already-computed multilingual records from multilingual_dict above.
    models_with_metadata = _GLOBAL_CACHE.available_models_with_metadata or []
    model_choices = create_model_choices(models_with_metadata)
    init_dict = {
        "multilingual_df": multilingual_dict["multilingual_df"],
        "multilingual_counts": multilingual_dict["multilingual_counts"],
        "lang_dfs": {},
        "lang_count_infos": {},
        "model_choices": [{"display": d, "value": v} for d, v in model_choices],
        "filter_state": {
            "selected_langs": default_filter_state.selected_langs,
            "selected_domains": default_filter_state.selected_domains,
            "selected_multilingual_domains": default_filter_state.selected_multilingual_domains,
        },
    }
    _GLOBAL_CACHE._json_leaderboard_init = json.dumps(init_dict, default=str, ensure_ascii=False).encode("utf-8")
    _GLOBAL_CACHE._gz_leaderboard_init = _gzip_compress(_GLOBAL_CACHE._json_leaderboard_init)

    print("API: pre-serialization complete.")


# ---------------------------------------------------------------------------
# FastAPI lifespan: heavy startup work happens here
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    global benchmark_dir, results_dir, benchmark_file_hashes, tokenizer_hash_cache
    global dataset_meta, language_tabs, code_domains, submission_worker

    print("API lifespan: downloading repos...")
    benchmark_dir, results_dir, benchmark_file_hashes, tokenizer_hash_cache = (
        download_repos_once()
    )
    dataset_meta = load_dataset_meta(benchmark_dir / "dataset_meta.yaml")

    print("API lifespan: initializing global cache...")
    if not _GLOBAL_CACHE.is_initialized:
        initialize_global_cache()

    # Build language_tabs
    _tabs = []
    for lang_code, lang_data in dataset_meta.items():
        if isinstance(lang_data, dict) and "pretty_name" in lang_data:
            domains_list = []
            for src in lang_data.get("domains", []):
                domains_list.append((src.get("pretty_name", src["name"]), src["name"]))
            _tabs.append(
                {"code": lang_code, "name": lang_data["pretty_name"], "domains": domains_list}
            )
    suggested_order = [
        "code", "en", "pt", "es", "fr", "it", "de", "pl", "ru", "ar", "zh", "ja", "ko", "vi",
    ]
    language_tabs[:] = sorted(
        _tabs,
        key=lambda x: suggested_order.index(x["code"])
        if x["code"] in suggested_order
        else len(suggested_order),
    )

    code_domains.clear()
    for lang_tab in language_tabs:
        if lang_tab["code"] == "code":
            for _, domain_name in lang_tab["domains"]:
                code_domains.add(domain_name)

    print("API lifespan: computing default filter result...")
    default_filter_state = _make_default_filter_state()
    default_compute_result = compute_leaderboard(default_filter_state)
    _GLOBAL_CACHE.default_filter_state = default_filter_state
    _GLOBAL_CACHE.default_compute_result = default_compute_result
    print("API lifespan: default filter result cached.")

    _preserialize_defaults(default_filter_state, default_compute_result)
    api_v2.configure(dataset_meta, _GLOBAL_CACHE.raw_results_df, _GLOBAL_CACHE._data_version)

    print("API lifespan: starting SubmissionWorker...")
    submission_worker = SubmissionWorker(
        benchmark_dir=benchmark_dir,
        results_dir=results_dir,
        benchmark_file_hashes=benchmark_file_hashes,
        tokenizer_hash_cache=tokenizer_hash_cache,
    )
    submission_worker.start()
    print("API lifespan: ready.")

    yield

    if submission_worker:
        submission_worker.stop()


# ---------------------------------------------------------------------------
# Response helper: serves pre-compressed bytes with ETag
# ---------------------------------------------------------------------------

def _cached_response(request: Request, raw: bytes, gzipped: bytes) -> Response:
    """Return pre-serialized JSON, choosing gzip if the client supports it.
    Includes ETag and Cache-Control headers for browser-side caching."""
    accept = request.headers.get("accept-encoding", "")
    use_gzip = "gzip" in accept and gzipped is not None
    headers: Dict[str, str] = {}
    if use_gzip:
        headers["Content-Encoding"] = "gzip"
    if _GLOBAL_CACHE._data_version:
        headers["ETag"] = f'W/"{APP_VERSION}:{_GLOBAL_CACHE._data_version}"'
        headers["Cache-Control"] = "public, max-age=300"
    return Response(
        content=gzipped if use_gzip else raw,
        media_type="application/json",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="African Tokenizer Leaderboard API", lifespan=lifespan)
app.include_router(api_v2.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)


@app.get("/api/health")
def health():
    return {"status": "ok", "initialized": _GLOBAL_CACHE.is_initialized}


@app.get("/api/metadata")
def metadata(request: Request):
    if _GLOBAL_CACHE._json_metadata is not None:
        print("cache hit: /api/metadata")
        return _cached_response(request, _GLOBAL_CACHE._json_metadata, _GLOBAL_CACHE._gz_metadata)
    raise HTTPException(status_code=503, detail="Metadata not yet initialized.")


@app.get("/api/leaderboard/init")
async def leaderboard_init(request: Request):
    """Lightweight initial payload: multilingual table + model_choices, no per-lang data.
    Lets the frontend show the main page immediately while language tabs load in background."""
    if _GLOBAL_CACHE._json_leaderboard_init is not None:
        print("cache hit: /api/leaderboard/init")
        return _cached_response(request, _GLOBAL_CACHE._json_leaderboard_init, _GLOBAL_CACHE._gz_leaderboard_init)
    raise HTTPException(status_code=503, detail="Leaderboard not yet initialized.")


@app.post("/api/leaderboard")
async def leaderboard(req: FilterRequest, request: Request):
    if await request.is_disconnected():
        return Response(status_code=499)

    filter_state = FilterState(
        selected_langs=req.selected_langs,
        selected_domains=req.selected_domains,
        selected_multilingual_domains=req.selected_multilingual_domains,
    )

    # Fast path: return pre-computed result for the default filter state (avoids re-computation)
    if (
        _GLOBAL_CACHE.default_compute_result is not None
        and _GLOBAL_CACHE.default_filter_state is not None
        and filter_state == _GLOBAL_CACHE.default_filter_state
        and req.limit == LIMIT_MODELS_SHOW
    ):
        print("cache hit: /api/leaderboard (default filter, using pre-computed result)")
        return _build_leaderboard_response(_GLOBAL_CACHE.default_compute_result, filter_state, req.limit)

    result = await asyncio.to_thread(compute_leaderboard, filter_state)
    return _build_leaderboard_response(result, filter_state, req.limit)


@app.post("/api/leaderboard/refresh")
async def leaderboard_refresh(request: Request):
    refresh_global_cache()

    default_filter_state = _make_default_filter_state()
    default_compute_result = await asyncio.to_thread(compute_leaderboard, default_filter_state)
    _GLOBAL_CACHE.default_filter_state = default_filter_state
    _GLOBAL_CACHE.default_compute_result = default_compute_result
    _preserialize_defaults(default_filter_state, default_compute_result)
    api_v2.configure(dataset_meta, _GLOBAL_CACHE.raw_results_df, _GLOBAL_CACHE._data_version)
    return _cached_response(request, _GLOBAL_CACHE._json_leaderboard_init, _GLOBAL_CACHE._gz_leaderboard_init)


@app.post("/api/leaderboard/lang/{lang_code}")
async def leaderboard_lang(lang_code: str, req: LangFilterRequest, request: Request):
    if await request.is_disconnected():
        return Response(status_code=499)

    # Fast path: return pre-serialized bytes when all domains for this lang are selected
    default_domains = _GLOBAL_CACHE._default_lang_domains.get(lang_code)
    if (
        default_domains is not None
        and lang_code in _GLOBAL_CACHE._json_lang_default
        and set(req.selected_domains) == default_domains
    ):
        print(f"cache hit: /api/leaderboard/lang/{lang_code} (default domains)")
        return _cached_response(request, _GLOBAL_CACHE._json_lang_default[lang_code], _GLOBAL_CACHE._gz_lang_default.get(lang_code))

    df, total_count, filtered_count = await asyncio.to_thread(
        _compute_single_lang, lang_code, req.selected_domains
    )
    return {
        "lang_df": _df_to_records(df),
        "lang_counts": {"total": total_count, "domain_filtered": filtered_count},
    }


@app.post("/api/leaderboard/multilingual")
async def leaderboard_multilingual(req: MultilingualFilterRequest, request: Request):
    if await request.is_disconnected():
        return Response(status_code=499)

    # Fast path: return pre-serialized bytes when all langs and domains are selected
    if (
        _GLOBAL_CACHE._json_multilingual_default is not None
        and set(req.selected_langs) == _GLOBAL_CACHE._default_multilingual_langs
        and set(req.selected_multilingual_domains) == _GLOBAL_CACHE._default_multilingual_domains
    ):
        print("cache hit: /api/leaderboard/multilingual (default filters)")
        return _cached_response(request, _GLOBAL_CACHE._json_multilingual_default, _GLOBAL_CACHE._gz_multilingual_default)

    df, total_count, filtered_count = await asyncio.to_thread(
        _compute_multilingual, req.selected_langs, req.selected_multilingual_domains
    )
    return {
        "multilingual_df": _df_to_records(df),
        "multilingual_counts": {"total": total_count, "domain_filtered": filtered_count},
    }


@app.post("/api/submit")
def submit_model(req: SubmitRequest):
    from transformers import AutoTokenizer

    if not req.model_name or not req.model_name.strip():
        raise HTTPException(status_code=400, detail="Model name cannot be empty.")

    revision = req.revision or "main"
    subfolder = req.subfolder if req.subfolder and req.subfolder.strip() else None
    selected_langs = req.selected_langs if req.selected_langs else None

    model_key_to_check = generate_model_key(
        req.model_name, revision, subfolder, req.target_mode
    )

    try:
        raw_data = _GLOBAL_CACHE.df_multilingual_raw
        if not raw_data.empty and "model_key" in raw_data.columns:
            if model_key_to_check in raw_data["model_key"].tolist():
                return {
                    "message": f"Model '{model_key_to_check}' has already been fully evaluated. Submission refused.",
                    "worker_status_df": _get_worker_status_records(),
                }

        current_worker_status = submission_worker.get_status()
        active = {
            m["model_key"]: m["status"]
            for m in current_worker_status["all_details"]
        }
        if model_key_to_check in active and active[model_key_to_check] in (
            "Queued",
            "Running",
        ):
            return {
                "message": f"Model '{model_key_to_check}' is already {active[model_key_to_check]}.",
                "worker_status_df": _get_worker_status_records(),
            }
    except Exception as e:
        print(f"Error during duplicate check: {e}")
        traceback.print_exc()

    try:
        subfolder_info = f", subfolder: {subfolder}" if subfolder else ""
        target_mode_info = ", target mode: enabled" if req.target_mode else ""
        print(
            f"Attempting to load tokenizer for: {req.model_name} "
            f"(revision: {revision}{subfolder_info}{target_mode_info})"
        )
        kwargs = {"trust_remote_code": False, "revision": revision}
        if subfolder:
            kwargs["subfolder"] = subfolder
        tokenizer = AutoTokenizer.from_pretrained(req.model_name, **kwargs)
        if req.target_mode:
            if hasattr(tokenizer, '_switch_to_target_mode'):
                tokenizer._switch_to_target_mode()
            elif hasattr(tokenizer, 'as_target_tokenizer'):
                tokenizer.as_target_tokenizer()
    except Exception as e:
        error_msg = (
            f"Error loading tokenizer for '{req.model_name}' "
            f"(revision: {revision}{subfolder_info}{target_mode_info}): "
            f"{type(e).__name__}: {e}"
        )
        return {"message": error_msg, "worker_status_df": _get_worker_status_records()}

    try:
        submission_worker.submit_model(
            req.model_name,
            revision=revision,
            subfolder=subfolder,
            _target_mode=req.target_mode,
            selected_langs=selected_langs,
        )
        lang_msg = (
            f"for languages: {', '.join(selected_langs)}"
            if selected_langs
            else "for all languages"
        )
        subfolder_msg = f" (subfolder: {subfolder})" if subfolder else ""
        target_mode_msg = " (target mode: enabled)" if req.target_mode else ""
        return {
            "message": (
                f"Model '{req.model_name}' (revision: {revision}{subfolder_msg}{target_mode_msg}) "
                f"added to queue {lang_msg}."
            ),
            "worker_status_df": _get_worker_status_records(),
        }
    except Exception as e:
        print(f"Error queueing model: {e}")
        traceback.print_exc()
        return {
            "message": f"Error queueing model: {e}",
            "worker_status_df": _get_worker_status_records(),
        }


def _get_worker_status_records():
    df_headers = ["Model Name", "Status", "Submitted At", "Error"]
    try:
        worker_status = submission_worker.get_status()
        all_model_details = worker_status.get("all_details", [])
        if all_model_details:
            temp_df = pd.DataFrame(all_model_details)
            temp_df = temp_df.rename(
                columns={
                    "model_name": "Model Name",
                    "revision": "Revision",
                    "subfolder": "Subfolder",
                    "_target_mode": "Target Mode",
                    "status": "Status",
                    "submission_time": "Submitted At",
                    "error": "Error",
                }
            )
            status_df = temp_df.reindex(columns=df_headers)
        else:
            status_df = pd.DataFrame(columns=df_headers)
        status_df["Error"] = status_df["Error"].fillna("")
        return _df_to_records(status_df)
    except Exception:
        traceback.print_exc()
        return []


@app.get("/api/worker/status")
def worker_status():
    return {"worker_status_df": _get_worker_status_records()}


@app.post("/api/battle/tokenize")
def battle_tokenize(req: BattleRequest):
    if not req.text:
        return {"error": "Please enter some text to tokenize.", "results": None}
    if not req.model_selection_1 or not req.model_selection_2:
        return {"error": "Please select both tokenizers.", "results": None}

    def _serialize_single(result_tuple):
        highlighted_text, count, vocab_size, token_df = result_tuple
        highlight_data = []
        color_map = {}
        if hasattr(highlighted_text, "value") and highlighted_text.value:
            highlight_data = highlighted_text.value
        if hasattr(highlighted_text, "color_map") and highlighted_text.color_map:
            color_map = highlighted_text.color_map
        return {
            "highlight_data": highlight_data,
            "color_map": color_map,
            "count": count,
            "vocab_size": vocab_size if vocab_size != "Error" else None,
            "table": _df_to_records(token_df) if isinstance(token_df, pd.DataFrame) else [],
        }

    r1 = _battle_tokenize_single(req.text, req.model_selection_1, req.better_tokenization)
    r2 = _battle_tokenize_single(req.text, req.model_selection_2, req.better_tokenization)

    return {
        "error": None,
        "model1": _serialize_single(r1),
        "model2": _serialize_single(r2),
    }


@app.post("/api/search/validate")
def validate_search(req: ValidateSearchRequest):
    if not req.query or not req.query.strip():
        return {"is_valid": True, "error_message": ""}
    is_valid, error_message = search_filter.validate_query(req.query)
    return {"is_valid": is_valid, "error_message": error_message}


@app.post("/api/domains/for-languages")
def domains_for_languages(req: DomainsForLanguagesRequest):
    domains = get_available_domains_for_languages(req.selected_lang_codes)
    return {"domains": domains}


# ---------------------------------------------------------------------------
# Content endpoints for React frontend
# ---------------------------------------------------------------------------

@app.get("/api/content/{page}")
def get_content(page: str):
    from display import LEADERBOARD_DESCRIPTION, OFFLINE_DOCS, ABOUT_MARKDOWN, MULTILINGUAL_DESCRIPTION
    content_map = {
        "about": ABOUT_MARKDOWN,
        "guide": OFFLINE_DOCS,
        "description": LEADERBOARD_DESCRIPTION,
        "multilingual": MULTILINGUAL_DESCRIPTION,
    }
    md_text = content_map.get(page)
    if md_text is None:
        raise HTTPException(status_code=404, detail="Content not found")
    return {"content": md_text}


# ---------------------------------------------------------------------------
# Serve React frontend static files (production)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

if _FRONTEND_DIR.is_dir():
    if (_FRONTEND_DIR / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_FRONTEND_DIR / "assets")),
            name="frontend-assets",
        )

    @app.get("/")
    async def serve_root():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))

    @app.get("/{catch_all:path}")
    async def serve_spa(catch_all: str):
        fp = _FRONTEND_DIR / catch_all
        if fp.is_file():
            return FileResponse(str(fp))
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
