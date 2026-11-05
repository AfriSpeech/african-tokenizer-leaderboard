import os
import pandas as pd
import numpy as np
from pathlib import Path
import time
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple
import json

from env import (
    DATA_DIR,
    DEFAULT_RESULTS_PATH,
    DEFAULT_MODEL_EVALUATIONS_PATH,
    ENABLE_LOAD_CACHE
)

from util import (
    ALL_COLUMNS,
    ensure_dataframe_has_columns,
)

RAW_RESULTS_FILE = 'raw_results.parquet'
MULTILINGUAL_OVERALL_RESULTS_FILE = 'multilingual_overall_results.parquet'
LANGUAGE_OVERALL_RESULTS_FILE = 'language_overall_results.parquet'
ENABLE_LOAD_CACHE = True

FREQ_COLS = ['freq_cardinality', 'freq_cardinality_ratio', 'freq_auc', 'freq_slope', 'freq_power_law',
             'freq_entropy', 'freq_renyi_entropy',
             'freq_shannon_efficiency', 'freq_renyi_efficiency',
             'freq_shannon_efficiency_full', 'freq_renyi_efficiency_full',
             'freq_percentile_freq',
             'freq_hapax_count', 'freq_hapax_ratio',
             'freq_half_mass_count', 'freq_90pct_mass_count', 'freq_90pct_mass_ratio', 'freq_gini']

# =============================================================================
# OPTIMIZED VECTORIZED CALCULATION FUNCTIONS FOR MULTI-MODEL PROCESSING
# =============================================================================

# Load all model results for comparison
def load_all_model_results(results_dir=None):
    """Load all model results for comparison."""
    if results_dir is None:
        results_dir = Path(DATA_DIR) / DEFAULT_RESULTS_PATH / DEFAULT_MODEL_EVALUATIONS_PATH
    
    data_dir = Path(DATA_DIR)
    evaluations_dir = results_dir / DEFAULT_MODEL_EVALUATIONS_PATH
    # Find all result files locally
    result_files = list(Path(evaluations_dir).glob("results_*.jsonl"))

    old_results_df = ensure_dataframe_has_columns(pd.DataFrame(), ALL_COLUMNS)
    old_results_df['_filepath'] = None

    if os.path.exists(data_dir / RAW_RESULTS_FILE) and ENABLE_LOAD_CACHE:
        old_results_df = pd.read_parquet(data_dir / RAW_RESULTS_FILE)
    
    already_processed_files = set(old_results_df['_filepath'].unique())
    print("load_all_model_results already_processed_files:", len(already_processed_files))
    # Load all result files
    all_results = []
    for file_path in tqdm(result_files, desc="Loading results files"):
        if str(file_path) in already_processed_files:
            continue
        #print('reading file_path', file_path)
        with open(file_path, 'r') as f:
            for line in f:
                domain_eval = json.loads(line)
                domain_eval['_filepath'] = file_path
                all_results.append(domain_eval)
        #df['_filepath'] = file_path
        # Ensure consistent columns
        #df = ensure_dataframe_has_columns(df, ALL_COLUMNS)
        #all_results.append(df)
    
    if len(all_results) > 0:
        all_df = pd.DataFrame(all_results)
        all_df = pd.concat([old_results_df, all_df], ignore_index=True)

        all_df = ensure_dataframe_has_columns(all_df, ALL_COLUMNS)
        
        if ENABLE_LOAD_CACHE:
            all_df['_filepath'] = all_df['_filepath'].fillna(pd.NA).astype(pd.StringDtype())
            all_df.to_parquet(data_dir / RAW_RESULTS_FILE)

        all_df = all_df.drop(columns=['_filepath'])

        return all_df
    
    old_results_df = old_results_df.drop(columns=['_filepath'])
    return old_results_df

def calculate_language_summaries_vectorized(df: pd.DataFrame, lookup_df: pd.DataFrame = None, language_rows_df: pd.DataFrame = None):
    """
    Calculate language overall summaries from raw evaluation data using vectorized operations.
    
    Args:
        df: Raw evaluation DataFrame (with evaluation_type='standard' rows)
        lookup_df: Optional DataFrame to lookup parity scores from
        language_rows_df: Optional DataFrame with evaluation_type='language' rows containing
            pre-computed Zipf metrics (freq_cardinality, freq_auc, freq_slope, freq_power_law).
            When provided, these values replace the per-domain means for those four columns.
            When None, the mean of per-domain values is used as fallback.
        
    Returns:
        DataFrame with language overall summaries (one row per model_key + lang combination)
    """
    if df.empty:
        return ensure_dataframe_has_columns(pd.DataFrame(), ALL_COLUMNS)
    
    # Ensure numeric columns are actually numeric (make a copy to avoid warnings)
    df = df.copy()
    standard_numeric_cols = [
        'total_tokens', 'total_words', 'total_chars', 'total_bytes',
        'total_docs', 'reversible_docs_count', 'unk_tokens_count',
    ]
    for col in standard_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Raw fidelity accumulators: keep NaN so downstream _compute_fidelity_metrics
    # correctly yields NaN (not zero) when a model has no experimental data.
    nullable_numeric_cols = [
        'byte_edit_distance_total', 'byte_edit_denominator_total', 'decoded_bytes_total',
        'char_edit_distance_total', 'char_edit_denominator_total', 'decoded_chars_total',
    ]
    for col in nullable_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure Zipf metric columns are numeric (may be missing in older rows).
    # Do NOT fillna(0): all-NaN groups should aggregate to NaN, not 0.
    for col in FREQ_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Get standard aggregations and add language-specific ones
    aggregations = _get_standard_aggregations()
    aggregations.update({
        # Sum the counts from all domains for each language
        'total_tokens': 'sum',
        'total_words': 'sum',
        'total_chars': 'sum',
        'total_bytes': 'sum',
        'total_docs': 'sum',
        'reversible_docs_count': 'sum',
        'unk_tokens_count': 'sum',
        'byte_edit_distance_total': 'sum',
        'byte_edit_denominator_total': 'sum',
        'decoded_bytes_total': 'sum',
        'char_edit_distance_total': 'sum',
        'char_edit_denominator_total': 'sum',
        'decoded_chars_total': 'sum',
    })
    # Fallback: mean of per-domain Zipf values (used when no language_rows_df)
    for col in FREQ_COLS:
        if col in df.columns:
            aggregations[col] = 'mean'
    
    # Perform the main aggregation - this replaces the Python for loop
    summary_df = df.groupby(['model_key', 'lang']).agg(aggregations).reset_index()
    
    # Calculate derived metrics using vectorized operations (much faster than loops)
    
    # Fertility: tokens per word
    fertility_ratio = summary_df['total_tokens'].divide(summary_df['total_words']).replace(np.inf, np.nan)
    summary_df['fertility'] = np.where(fertility_ratio > 0, fertility_ratio.round(3), 0)
    summary_df['fertility'] = summary_df['fertility'].fillna(0)
    
    # Compression chars: chars per token
    compression_chars_ratio = summary_df['total_chars'].divide(summary_df['total_tokens']).replace(np.inf, np.nan)
    summary_df['compression_chars'] = np.where(compression_chars_ratio > 0, compression_chars_ratio.round(3), 0)
    summary_df['compression_chars'] = summary_df['compression_chars'].fillna(0)
    
    # Compression bytes: bytes per token
    compression_bytes_ratio = summary_df['total_bytes'].divide(summary_df['total_tokens']).replace(np.inf, np.nan)
    summary_df['compression_bytes'] = np.where(compression_bytes_ratio > 0, compression_bytes_ratio.round(3), 0)
    summary_df['compression_bytes'] = summary_df['compression_bytes'].fillna(0)
    
    # Reversible ratio: reversible docs / total docs
    reversible_ratio = summary_df['reversible_docs_count'].divide(summary_df['total_docs']).replace(np.inf, np.nan)
    summary_df['reversible_ratio'] = np.where(reversible_ratio > 0, reversible_ratio.round(4), 0)
    summary_df['reversible_ratio'] = summary_df['reversible_ratio'].fillna(0)
    
    # Unknown ratio: unk tokens / total tokens
    unk_ratio = summary_df['unk_tokens_count'].divide(summary_df['total_tokens']).replace(np.inf, np.nan)
    summary_df['unk_ratio'] = np.where(unk_ratio > 0, unk_ratio.round(5), 0)
    summary_df['unk_ratio'] = summary_df['unk_ratio'].fillna(0)

    summary_df = _compute_fidelity_metrics(summary_df)
    
    # Override Zipf metrics with pre-computed language-level values when available.
    # These were computed from the merged frequency distribution of all domains and are
    # more accurate than averaging per-domain values.
    if language_rows_df is not None and not language_rows_df.empty:
        freq_source = language_rows_df.copy()
        for col in FREQ_COLS:
            if col in freq_source.columns:
                freq_source[col] = pd.to_numeric(freq_source[col], errors='coerce')
        merge_cols = ['model_key', 'lang']
        available_freq_cols = [c for c in FREQ_COLS if c in freq_source.columns]
        if available_freq_cols:
            freq_df = freq_source[merge_cols + available_freq_cols].copy()
            # Drop fallback means then re-attach real language-level values
            summary_df = summary_df.drop(columns=available_freq_cols, errors='ignore')
            summary_df = summary_df.merge(freq_df, on=merge_cols, how='left')
            # Models without a language row keep NaN (displayed as "—" in UI).
            summary_df = _compute_fidelity_metrics(summary_df)

    # Add domain and evaluation_type for consistency
    summary_df['domain'] = 'lang_overall'
    summary_df['evaluation_type'] = 'standard'
    summary_df['reference'] = False
    
    # Handle parity scores by merging with lookup_df if provided
    if lookup_df is not None and not lookup_df.empty:
        # Find parity columns in lookup_df
        parity_cols = [col for col in lookup_df.columns if col.startswith('parity_')]
        if parity_cols:
            # Merge to attach parity scores
            merge_cols = ['model_key', 'lang']
            merge_df = lookup_df[merge_cols + parity_cols].copy()
            summary_df = summary_df.merge(merge_df, on=merge_cols, how='left')
    
    return ensure_dataframe_has_columns(summary_df, ALL_COLUMNS)


def calculate_multilingual_summaries_vectorized(lang_summaries_df: pd.DataFrame, multilingual_rows_df: pd.DataFrame = None):
    """
    Calculate multilingual overall summaries from language summaries using vectorized operations.
    
    Args:
        lang_summaries_df: Language overall summaries DataFrame
        multilingual_rows_df: Optional DataFrame with evaluation_type='multilingual' rows.
            When provided, Zipf metrics and all standard metrics are taken from these
            pre-computed rows (which were computed from the full merged token distribution)
            rather than being averaged/summed from per-language summaries.
        
    Returns:
        DataFrame with multilingual overall summaries (one row per model_key)
    """
    if lang_summaries_df.empty:
        return ensure_dataframe_has_columns(pd.DataFrame(), ALL_COLUMNS)
    
    # Ensure numeric columns are actually numeric (make a copy to avoid warnings).
    # Standard (non-experimental) cols get fillna(0); experimental nullable cols keep NaN.
    lang_summaries_df = lang_summaries_df.copy()
    standard_numeric_cols = [
        'fertility', 'compression_chars', 'compression_bytes', 'reversible_ratio', 'unk_ratio',
        'total_tokens', 'total_words', 'total_chars', 'total_bytes',
        'total_docs', 'reversible_docs_count', 'unk_tokens_count',
    ]
    nullable_numeric_cols = [
        'byte_fidelity', 'effective_bytes_per_token', 'ebpb', 'ebpb_full',
        'char_fidelity', 'effective_chars_per_token', 'ebpc',
        'byte_edit_distance_total', 'byte_edit_denominator_total', 'decoded_bytes_total',
        'char_edit_distance_total', 'char_edit_denominator_total', 'decoded_chars_total',
    ]
    for col in standard_numeric_cols:
        if col in lang_summaries_df.columns:
            lang_summaries_df[col] = pd.to_numeric(lang_summaries_df[col], errors='coerce').fillna(0)
    for col in nullable_numeric_cols:
        if col in lang_summaries_df.columns:
            lang_summaries_df[col] = pd.to_numeric(lang_summaries_df[col], errors='coerce')
    
    # Get standard aggregations and add multilingual-specific ones
    aggregations = _get_standard_aggregations()
    aggregations.update({
        # For multilingual, we AVERAGE the per-language metrics
        'fertility': 'mean',
        'compression_chars': 'mean',
        'compression_bytes': 'mean',
        'reversible_ratio': 'mean',
        'unk_ratio': 'mean',
        
        # But we still sum the total counts
        'total_tokens': 'sum',
        'total_words': 'sum',
        'total_chars': 'sum',
        'total_bytes': 'sum',
        'total_docs': 'sum',
        'reversible_docs_count': 'sum',
        'unk_tokens_count': 'sum',
        'byte_edit_distance_total': 'sum',
        'byte_edit_denominator_total': 'sum',
        'decoded_bytes_total': 'sum',
        'char_edit_distance_total': 'sum',
        'char_edit_denominator_total': 'sum',
        'decoded_chars_total': 'sum',
    })
    
    # Add parity columns if they exist
    parity_cols = [col for col in lang_summaries_df.columns if col.startswith('parity_')]
    for col in parity_cols:
        aggregations[col] = 'mean'
    
    # Mean of per-language Zipf metrics — keep NaN for models without freq data.
    for col in FREQ_COLS:
        if col in lang_summaries_df.columns:
            lang_summaries_df[col] = pd.to_numeric(lang_summaries_df[col], errors='coerce')
            aggregations[col] = 'mean'
    
    # Perform the aggregation - this replaces the Python for loop
    summary_df = lang_summaries_df.groupby('model_key').agg(aggregations).reset_index()

    summary_df = _compute_fidelity_metrics(summary_df)
    
    # Apply standard rounding to averaged metrics
    summary_df = _apply_metric_rounding(summary_df, parity_cols)
    
    # Override all metrics with pre-computed multilingual-row values when available.
    # These were derived from the full global token distribution, not averaged from languages.
    if multilingual_rows_df is not None and not multilingual_rows_df.empty:
        override_cols = FREQ_COLS + [
            'fertility', 'compression_chars', 'compression_bytes',
            'reversible_ratio', 'unk_ratio', 'byte_fidelity',
            'effective_bytes_per_token', 'ebpb', 'ebpb_full',
            'char_fidelity', 'effective_chars_per_token', 'ebpc',
            'total_tokens', 'total_words', 'total_chars', 'total_bytes',
            'total_docs', 'reversible_docs_count', 'unk_tokens_count',
            'byte_edit_distance_total', 'byte_edit_denominator_total', 'decoded_bytes_total',
            'char_edit_distance_total', 'char_edit_denominator_total', 'decoded_chars_total',
        ]
        src = multilingual_rows_df.copy()
        for col in override_cols:
            if col in src.columns:
                src[col] = pd.to_numeric(src[col], errors='coerce')
        available_override = [c for c in override_cols if c in src.columns]
        if available_override:
            override_df = src[['model_key'] + available_override].copy()
            summary_df = summary_df.drop(columns=available_override, errors='ignore')
            summary_df = summary_df.merge(override_df, on='model_key', how='left')
            # Experimental cols stay NaN for models not in multilingual_rows_df.
            summary_df = _compute_fidelity_metrics(summary_df)

    # Add fixed fields for multilingual summaries
    summary_df['lang'] = 'all'
    summary_df['domain'] = 'multilingual_overall'
    summary_df['evaluation_type'] = 'standard'
    summary_df['reference'] = False
    
    return ensure_dataframe_has_columns(summary_df, ALL_COLUMNS)


def calculate_summaries_vectorized(df: pd.DataFrame, is_multilingual: bool = False, 
                                  lookup_df: pd.DataFrame = None,
                                  language_rows_df: pd.DataFrame = None,
                                  multilingual_rows_df: pd.DataFrame = None):
    """
    Unified function to calculate either language or multilingual summaries using vectorized operations.
    This function replaces all the slow Python loops in both get_overall_summary and apply_all_filters.
    
    Args:
        df: Input DataFrame (raw evaluation data for language summaries, 
            language summaries for multilingual summaries)
        is_multilingual: If True, calculates multilingual summaries from language summaries.
                        If False, calculates language summaries from raw data.
        lookup_df: Optional DataFrame for parity score lookup (only used for language summaries)
        language_rows_df: Optional DataFrame with evaluation_type='language' rows for Zipf
            metric lookup (only used for full-language summaries, not filtered-domain paths).
        multilingual_rows_df: Optional DataFrame with evaluation_type='multilingual' rows for
            overriding computed multilingual metrics with pre-computed values.
        
    Returns:
        DataFrame with calculated summaries
    """
    if is_multilingual:
        return calculate_multilingual_summaries_vectorized(df, multilingual_rows_df=multilingual_rows_df)
    else:
        return calculate_language_summaries_vectorized(df, lookup_df, language_rows_df=language_rows_df)


def calculate_parity_scores_vectorized(df_parity: pd.DataFrame):
    """
    Calculate parity scores from parity evaluation data using vectorized operations.
    
    Args:
        df_parity: DataFrame with evaluation_type='parity' rows
        
    Returns:
        DataFrame with parity scores indexed by model_key, lang, and parity domain
    """
    if df_parity.empty:
        return pd.DataFrame()
    
    # Filter for non-reference parity data (the model evaluations)
    model_parity_df = df_parity[~df_parity['reference']].copy()
    
    # Filter for reference parity data
    reference_parity_df = df_parity[df_parity['reference']].copy()
    
    if model_parity_df.empty or reference_parity_df.empty:
        return pd.DataFrame()
    
    # Reference tokens per (tokenizer, parity set): each tokenizer splits the reference text
    # differently, and a benchmark may give each language its own parallel reference
    # ("parity_en__twi": Twi verses vs the same verses in English). Keys without model_key
    # fall back to the parity set alone (one shared reference file, as with FLORES).
    if 'model_key' in df_parity.columns and df_parity['model_key'].notna().all():
        reference_lookup = (
            reference_parity_df.groupby(['model_key', 'domain'])['total_tokens'].first().to_dict()
        )
        model_parity_df['reference_tokens'] = [
            reference_lookup.get((k, d)) for k, d in
            zip(model_parity_df['model_key'], model_parity_df['domain'])
        ]
    else:
        reference_lookup = reference_parity_df.groupby('domain')['total_tokens'].first().to_dict()
        model_parity_df['reference_tokens'] = model_parity_df['domain'].map(reference_lookup)
    model_parity_df['reference_tokens'] = pd.to_numeric(
        model_parity_df['reference_tokens'], errors='coerce'
    )
    
    # Calculate parity ratio (model_tokens / reference_tokens)
    parity_ratio = model_parity_df['total_tokens'].divide(model_parity_df['reference_tokens']).replace(np.inf, np.nan)
    model_parity_df['parity_value'] = np.where(parity_ratio > 0, parity_ratio.round(3), 0)
    model_parity_df['parity_value'] = model_parity_df['parity_value'].fillna(0)
    
    # Add 'parity_' prefix to domain names if not already present
    # "parity_en__twi" (a per-language parity set) reports as the "parity_en" column.
    model_parity_df['parity_column'] = model_parity_df['domain'].apply(
        lambda x: (lambda base: base if base.startswith('parity_') else f'parity_{base}')(
            x.split('__', 1)[0]
        )
    )
    
    # Pivot to create parity columns
    parity_scores_df = model_parity_df.pivot_table(
        index=['model_key', 'lang'], 
        columns='parity_column', 
        values='parity_value',
        aggfunc='first'  # Should be only one value per combination
    ).reset_index()
    
    # Flatten column names
    parity_scores_df.columns.name = None
    
    return parity_scores_df


def calculate_overall_for_model_vectorized(df, meta=None):
    """
    Calculate various overall metrics for a model's results using vectorized operations.
    This is the optimized version that replaces the slow Python loops with fast pandas operations.
    
    Args:
        df: DataFrame with model results
        tokenizer_meta: Optional tokenizer metadata
        meta: Optional dataset metadata from dataset_meta.yaml to calculate multilingual overalls
        
    Returns:
        Dictionary with lang_overall entries for each language and 
        potentially a multilingual_overall entry if all languages were evaluated
    """
    if df.empty:
        return {}
    
    # Ensure dataframe has all required columns
    df = ensure_dataframe_has_columns(df, ALL_COLUMNS)
    
    # Separate evaluation types
    df_parity = df[df['evaluation_type'] == 'parity'].copy()
    df_standard = df[df['evaluation_type'] == 'standard'].copy()
    df_language = df[df['evaluation_type'] == 'language'].copy()
    df_multilingual = df[df['evaluation_type'] == 'multilingual'].copy()
    
    if df_standard.empty:
        return {}
    
    # Calculate parity scores first (if any parity data exists)
    parity_lookup_df = None
    if not df_parity.empty:
        parity_lookup_df = calculate_parity_scores_vectorized(df_parity)
    
    # Calculate language overall summaries using vectorized operations
    lang_overalls_df = calculate_summaries_vectorized(
        df_standard, 
        is_multilingual=False, 
        lookup_df=parity_lookup_df,
        language_rows_df=df_language if not df_language.empty else None,
    )
    
    if lang_overalls_df.empty:
        return {}
    
    # Convert back to list of dictionaries for compatibility
    lang_overalls = lang_overalls_df.to_dict('records')
    
    # Check if we should calculate multilingual overall
    multilingual_overall = None
    if meta is not None:
        all_meta_langs = set(meta.keys())
        evaluated_langs = set(lang_overalls_df['lang'].unique())
        
        # Check if all languages in meta were evaluated
        if all_meta_langs.issubset(evaluated_langs):
            # Calculate multilingual overall using vectorized operations
            multilingual_df = calculate_summaries_vectorized(
                lang_overalls_df, 
                is_multilingual=True,
                multilingual_rows_df=df_multilingual if not df_multilingual.empty else None,
            )
            
            if not multilingual_df.empty:
                multilingual_overall = multilingual_df.iloc[0].to_dict()
    
    # Return the same structure as the original function
    result = {"lang_overalls": lang_overalls}
    if multilingual_overall is not None:
        result["multilingual_overall"] = multilingual_overall
    
    return result





# =============================================================================
# MULTI-MODEL RESULTS AGGREGATION FUNCTIONS
# =============================================================================

def _get_standard_aggregations():
    """Get standard aggregation operations for summary calculations."""
    return {
        # Metadata columns (take the first value from each group)
        'model': 'first',
        'revision': 'first',
        'subfolder': 'first',
        '_target_mode': 'first',
        'vocab_size': 'first',
        'tokenizer_class': 'first',
        'tokenizer_algorithm': 'first',
        'tokenizer_implementation': 'first',
        'vocab_hash': 'first',
        'tokenizer_config_hash': 'first',
        'vocab_near_duplicates': 'first',
        'likes': 'first',
        'created_at': 'first',
        'sha': 'first',
        'downloads_all_time': 'first',
        'author_type': 'first',
        'author_fullname': 'first',
        'author_followers': 'first',
        'author_models': 'first',
        'author_is_verified': 'first',
        'evaluation_date': 'max',  # Get the most recent evaluation date
    }

def _apply_metric_rounding(df, parity_cols=None):
    """Apply standard rounding to calculated metrics."""
    if parity_cols is None:
        parity_cols = [col for col in df.columns if col.startswith('parity_')]
    
    # Round the metrics to match original precision
    for col in ['fertility', 'compression_chars', 'compression_bytes', 'effective_bytes_per_token', 'ebpb', 'ebpb_full', 'effective_chars_per_token', 'ebpc']:
        if col in df.columns:
            df[col] = df[col].round(3)
    for col in ['reversible_ratio']:
        if col in df.columns:
            df[col] = df[col].round(4)
    for col in ['unk_ratio', 'byte_fidelity', 'char_fidelity']:
        if col in df.columns:
            df[col] = df[col].round(5)
    for col in parity_cols:
        if col in df.columns:
            df[col] = df[col].round(3)
    for col in FREQ_COLS:
        if col in df.columns:
            df[col] = df[col].round(5)
    
    return df

def _compute_fidelity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute byte- and char-fidelity-derived metrics from aggregated raw totals.

    When the denominator column is missing or all-NaN for a row, the derived
    metrics stay NaN (displayed as "—" in the UI) instead of being coerced to 0.
    """
    df = df.copy()

    # Use .where() so that a zero denominator → NaN instead of inf/0.
    byte_denom = df['byte_edit_denominator_total'].where(df['byte_edit_denominator_total'] > 0)
    byte_fidelity_ratio = (1 - df['byte_edit_distance_total'].divide(byte_denom)).clip(lower=0)
    df['byte_fidelity'] = byte_fidelity_ratio  # NaN propagates when denom is NaN/0

    effective_bytes_ratio = (
        df['total_bytes'].divide(df['total_tokens']).replace(np.inf, np.nan) * df['byte_fidelity']
    ).clip(lower=0)
    df['effective_bytes_per_token'] = effective_bytes_ratio

    observed_vocab = df['freq_cardinality'].where(df['freq_cardinality'] > 0).clip(lower=2)
    byte_original_denom = df['total_bytes'].where(df['total_bytes'] > 0)
    ebpb_ratio = (
        df['total_tokens'] * np.log2(observed_vocab) + 8 * df['byte_edit_distance_total']
    ).divide(byte_original_denom)
    df['ebpb'] = ebpb_ratio.clip(lower=0)

    full_vocab = df['vocab_size'].where(df['vocab_size'] > 0).clip(lower=2)
    ebpb_full_ratio = (
        df['total_tokens'] * np.log2(full_vocab) + 8 * df['byte_edit_distance_total']
    ).divide(byte_original_denom)
    df['ebpb_full'] = ebpb_full_ratio.clip(lower=0)

    char_denom = df['char_edit_denominator_total'].where(df['char_edit_denominator_total'] > 0)
    char_fidelity_ratio = (1 - df['char_edit_distance_total'].divide(char_denom)).clip(lower=0)
    df['char_fidelity'] = char_fidelity_ratio

    effective_chars_ratio = (
        df['total_chars'].divide(df['total_tokens']).replace(np.inf, np.nan) * df['char_fidelity']
    ).clip(lower=0)
    df['effective_chars_per_token'] = effective_chars_ratio

    ebpc_ratio = np.log2(df['vocab_size'].clip(lower=2)).divide(
        df['effective_chars_per_token'].where(df['effective_chars_per_token'] > 0)
    )
    df['ebpc'] = ebpc_ratio.clip(lower=0)

    return df

def calculate_language_overalls_for_filters(raw_results_df, selected_domains, selected_languages, expected_columns):
    """
    Efficiently calculate language overalls for selected domains and languages using vectorized operations.
    Maintains language-weighted consistency for multilingual calculations.
    
    Args:
        raw_results_df: Raw evaluation DataFrame
        selected_domains: List of domain names to include
        selected_languages: List of language codes to include
        expected_columns: List of expected column names for output
        
    Returns:
        DataFrame with language overall results for selected domains and languages
    """
    if raw_results_df.empty or not selected_domains or not selected_languages:
        return ensure_dataframe_has_columns(pd.DataFrame(), expected_columns)
    
    # Filter raw data to selected domains and languages
    filtered_raw_df = raw_results_df[
        (raw_results_df['lang'].isin(selected_languages)) &
        (raw_results_df['domain'].isin(selected_domains)) &
        (raw_results_df['domain'] != 'lang_overall') &
        (raw_results_df['domain'] != 'multilingual_overall') &
        (raw_results_df['evaluation_type'] == 'standard')
    ].copy()
    
    if filtered_raw_df.empty:
        return ensure_dataframe_has_columns(pd.DataFrame(), expected_columns)
    
    # Calculate language overalls using existing vectorized function
    # This aggregates selected domains for each language
    lang_overalls = calculate_summaries_vectorized(
        filtered_raw_df, 
        is_multilingual=False, 
        lookup_df=None  # Skip parity for domain filtering - will be inconsistent anyway
    )
    
    return lang_overalls


def calculate_language_overalls_for_domains(raw_results_df, selected_domains, all_languages, expected_columns):
    """
    Efficiently calculate language overalls for selected domains using vectorized operations.
    Maintains language-weighted consistency for multilingual calculations.
    
    DEPRECATED: Use calculate_language_overalls_for_filters instead.
    
    Args:
        raw_results_df: Raw evaluation DataFrame
        selected_domains: List of domain names to include
        all_languages: List of all language codes to process
        expected_columns: List of expected column names for output
        
    Returns:
        DataFrame with language overall results for selected domains
    """
    return calculate_language_overalls_for_filters(raw_results_df, selected_domains, all_languages, expected_columns)


def get_overall_summary(benchmark_dir, results_dir, all_df=None):
    """
    Get overall summary table for all models using optimized vectorized operations.
    
    Args:
        benchmark_dir: Path to benchmark directory
        results_dir: Path to results directory  
        all_df: Optional pre-loaded DataFrame of all model results
        
    Returns:
        Dictionary with 'lang_overalls' and 'multilingual_overall' DataFrames
    """
    general_time = time.time()
    if all_df is None:
        all_df = load_all_model_results(results_dir)
    if all_df.empty:
        return {}
    
    # First, download metadata to calculate multilingual overall correctly
    data_dir = Path(DATA_DIR)
    dataset_dir = Path(benchmark_dir)
    meta_path = dataset_dir / 'dataset_meta.yaml'
    
    from tokenizer_evaluate import load_dataset_meta
    meta = load_dataset_meta(meta_path) if meta_path.exists() else None
    
    # Load existing cached results
    old_multilingual_df = ensure_dataframe_has_columns(pd.DataFrame(), ALL_COLUMNS)
    old_lang_overall_df = ensure_dataframe_has_columns(pd.DataFrame(), ALL_COLUMNS)
    if ENABLE_LOAD_CACHE:
        if os.path.exists(data_dir / MULTILINGUAL_OVERALL_RESULTS_FILE):
            old_multilingual_df = pd.read_parquet(data_dir / MULTILINGUAL_OVERALL_RESULTS_FILE)
        if os.path.exists(data_dir / LANGUAGE_OVERALL_RESULTS_FILE):
            old_lang_overall_df = pd.read_parquet(data_dir / LANGUAGE_OVERALL_RESULTS_FILE)
    
    already_processed_model_keys = set(old_multilingual_df['model_key'].unique())

    # Filter to only process new models (huge speedup)
    new_models_df = all_df[~all_df['model_key'].isin(already_processed_model_keys)]
    
    if new_models_df.empty:
        # No new models to process, return existing data
        return {
            "lang_overalls": old_lang_overall_df.sort_values(['lang', 'fertility'], ascending=[True, True]) if not old_lang_overall_df.empty else pd.DataFrame(columns=ALL_COLUMNS),
            "multilingual_overall": old_multilingual_df.sort_values('fertility', ascending=True) if not old_multilingual_df.empty else pd.DataFrame(columns=ALL_COLUMNS)
        }
    
    print(f"Processing {len(new_models_df['model_key'].unique())} new models using vectorized calculations...")
    
    # VECTORIZED APPROACH: Process all new models at once instead of one by one
    
    # Separate evaluation types for all new models
    df_parity = new_models_df[new_models_df['evaluation_type'] == 'parity'].copy()
    df_standard = new_models_df[new_models_df['evaluation_type'] == 'standard'].copy()
    df_language = new_models_df[new_models_df['evaluation_type'] == 'language'].copy()
    df_multilingual = new_models_df[new_models_df['evaluation_type'] == 'multilingual'].copy()
    
    new_lang_overall_results = []
    new_multilingual_overall_results = []
    
    if not df_standard.empty:
        # Calculate parity scores for all models at once
        parity_lookup_df = None
        if not df_parity.empty:
            parity_lookup_df = calculate_parity_scores_vectorized(df_parity)
        
        # Calculate language overall summaries for ALL models using vectorized operations
        all_lang_overalls_df = calculate_summaries_vectorized(
            df_standard, 
            is_multilingual=False, 
            lookup_df=parity_lookup_df,
            language_rows_df=df_language if not df_language.empty else None,
        )
        
        if not all_lang_overalls_df.empty:
            new_lang_overall_results = all_lang_overalls_df.to_dict('records')
            
            # Calculate multilingual overalls for models that have all languages
            if meta is not None:
                all_meta_langs = set(meta.keys())
                
                # VECTORIZED APPROACH: Filter to only include models that have all required languages
                model_lang_counts = all_lang_overalls_df.groupby('model_key')['lang'].nunique()
                models_with_all_langs = model_lang_counts[model_lang_counts == len(all_meta_langs)].index
                
                complete_lang_overalls = all_lang_overalls_df[
                    all_lang_overalls_df['model_key'].isin(models_with_all_langs)
                ].copy()
                
                if not complete_lang_overalls.empty:
                    # Calculate multilingual overalls for ALL qualifying models at once
                    multilingual_df_all = calculate_summaries_vectorized(
                        complete_lang_overalls, 
                        is_multilingual=True,
                        multilingual_rows_df=df_multilingual if not df_multilingual.empty else None,
                    )
                    
                    if not multilingual_df_all.empty:
                        new_multilingual_overall_results = multilingual_df_all.to_dict('records')
    
    # Combine with existing results
    lang_overall_df = old_lang_overall_df
    multilingual_df = old_multilingual_df
    
    if len(new_lang_overall_results) > 0:
        _lang_overall_df = pd.DataFrame(new_lang_overall_results)
        lang_overall_df = pd.concat([lang_overall_df, _lang_overall_df], ignore_index=True)
        
    if len(new_multilingual_overall_results) > 0:
        _multilingual_df = pd.DataFrame(new_multilingual_overall_results)
        multilingual_df = pd.concat([multilingual_df, _multilingual_df], ignore_index=True)

    # Ensure columns and save
    if len(new_lang_overall_results) > 0 or len(new_multilingual_overall_results) > 0:
        lang_overall_df = ensure_dataframe_has_columns(lang_overall_df, ALL_COLUMNS)
        multilingual_df = ensure_dataframe_has_columns(multilingual_df, ALL_COLUMNS)
        if ENABLE_LOAD_CACHE:
            lang_overall_df.to_parquet(data_dir / LANGUAGE_OVERALL_RESULTS_FILE)
            multilingual_df.to_parquet(data_dir / MULTILINGUAL_OVERALL_RESULTS_FILE)
        
        print(f"Processed {len(new_lang_overall_results)} language overalls and {len(new_multilingual_overall_results)} multilingual overalls using vectorized operations")

    time_total = time.time() - general_time
    print(f"Get overall summary total time: {time_total:.2f} seconds")

    return {
        "lang_overalls": lang_overall_df.sort_values(['lang', 'fertility'], ascending=[True, True]) if not lang_overall_df.empty else pd.DataFrame(columns=ALL_COLUMNS),
        "multilingual_overall": multilingual_df.sort_values('fertility', ascending=True) if not multilingual_df.empty else pd.DataFrame(columns=ALL_COLUMNS)
    } 