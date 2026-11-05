import json

import numpy as np
import pandas as pd

# =============================================================================
# COLUMN DEFINITIONS - CENTRALIZED MANAGEMENT
# =============================================================================

# The single domain of truth for all possible columns in a results DataFrame.
ALL_COLUMNS = [
    'model_key', 'model', 'revision', 'subfolder', '_target_mode', 'lang', 'domain', 'evaluation_type',
    'likes', 'created_at', 'fertility', 'compression_chars', 'compression_bytes', 'reversible_ratio', 'unk_ratio',
    'byte_fidelity', 'effective_bytes_per_token', 'ebpb', 'ebpb_full',
    'char_fidelity', 'effective_chars_per_token', 'ebpc',
    'total_tokens', 'total_words', 'total_chars', 'total_bytes', 'total_docs', 
    'reversible_docs_count', 'unk_tokens_count',
    'byte_edit_distance_total', 'byte_edit_denominator_total', 'decoded_bytes_total',
    'char_edit_distance_total', 'char_edit_denominator_total', 'decoded_chars_total',
    'freq_cardinality', 'freq_cardinality_ratio', 'freq_auc', 'freq_slope', 'freq_power_law',
    'freq_entropy', 'freq_renyi_entropy',
    'freq_shannon_efficiency', 'freq_renyi_efficiency',
    'freq_shannon_efficiency_full', 'freq_renyi_efficiency_full',
    'freq_percentile_freq',
    'freq_hapax_count', 'freq_hapax_ratio',
    'freq_half_mass_count', 'freq_90pct_mass_count', 'freq_90pct_mass_ratio', 'freq_gini',
    'vocab_size', 'vocab_hash', 'vocab_prefix', 'vocab_near_duplicates', 
    'vocab_count_duplicates_by_type', 'tokenizer_class',
    'tokenizer_algorithm', 'tokenizer_implementation',
    'tokenizer_config_hash', #'tokenizer_config',
    'benchmark_file_hash', 'evaluation_date', #'evaluation_time',
    'reference', 'sha', 'downloads_all_time',
    'author_type', 'author_fullname', 'author_followers', 'author_models', 'author_is_verified',
    # Parity columns like 'parity_en' are added dynamically.
]

# Columns that should default to 0 if missing.
NUMERIC_ZERO_DEFAULT_COLUMNS = [
    'fertility', 'compression_chars', 'compression_bytes', 'reversible_ratio', 'unk_ratio',
    'total_tokens', 'total_words', 'total_chars', 'total_bytes', 'total_docs',
    'reversible_docs_count', 'unk_tokens_count',
    'likes', 'downloads_all_time',
    'vocab_size', 'vocab_near_duplicates', # 'evaluation_time'
    'author_followers', 'author_models',
    # Parity columns are added dynamically.
]

# Experimental metrics — kept as NaN when missing rather than defaulting to 0.
# This lets the frontend display "—" and keeps nulls at the table bottom on sort.
NUMERIC_NULLABLE_COLUMNS = [
    # Fidelity (byte-level)
    'byte_fidelity', 'effective_bytes_per_token', 'ebpb', 'ebpb_full',
    # Fidelity (char-level)
    'char_fidelity', 'effective_chars_per_token', 'ebpc',
    # Raw edit-distance accumulators (used to recompute fidelity during aggregation)
    'byte_edit_distance_total', 'byte_edit_denominator_total', 'decoded_bytes_total',
    'char_edit_distance_total', 'char_edit_denominator_total', 'decoded_chars_total',
    # Token frequency / Zipf distribution metrics
    'freq_cardinality', 'freq_cardinality_ratio', 'freq_auc', 'freq_slope', 'freq_power_law',
    'freq_entropy', 'freq_renyi_entropy',
    'freq_shannon_efficiency', 'freq_renyi_efficiency',
    'freq_shannon_efficiency_full', 'freq_renyi_efficiency_full',
    'freq_percentile_freq',
    'freq_hapax_count', 'freq_hapax_ratio',
    'freq_half_mass_count', 'freq_90pct_mass_count', 'freq_90pct_mass_ratio', 'freq_gini',
]

# Columns that should default to a specific boolean value.
BOOLEAN_DEFAULT_COLUMNS = {
    'reference': False,
    '_target_mode': False,
    'author_is_verified': False,
}

DICT_TYPE_COLUMNS = ['vocab_count_duplicates_by_type']

PARITY_COLUMNS = ['parity_en']

def add_parity_column(column_name):
    """Dynamically add a parity column to the global definitions."""
    global ALL_COLUMNS, NUMERIC_ZERO_DEFAULT_COLUMNS
    
    if column_name not in ALL_COLUMNS:
        ALL_COLUMNS.append(column_name)
        if column_name not in NUMERIC_ZERO_DEFAULT_COLUMNS:
            NUMERIC_ZERO_DEFAULT_COLUMNS.append(column_name)

# Add the default parity column to ensure it's always there
for column in PARITY_COLUMNS:
    add_parity_column(column)


def ensure_dataframe_has_columns(df, required_columns=None, numeric_zero_cols=None, boolean_defaults=None, dict_type_cols=None):
    """
    Ensures a DataFrame has all required columns with appropriate defaults.
    
    Args:
        df: Input DataFrame (can be None or empty)
        required_columns: List of columns that must exist (defaults to ALL_COLUMNS)
        numeric_zero_cols: List of numeric columns that default to 0 (defaults to NUMERIC_ZERO_DEFAULT_COLUMNS)
        boolean_defaults: Dict of boolean columns and their defaults (defaults to BOOLEAN_DEFAULT_COLUMNS)
    
    Returns:
        DataFrame with all required columns and proper defaults
    """
    if df is None:
        df = pd.DataFrame()
    
    if required_columns is None:
        required_columns = ALL_COLUMNS
    if numeric_zero_cols is None:
        numeric_zero_cols = NUMERIC_ZERO_DEFAULT_COLUMNS
    if boolean_defaults is None:
        boolean_defaults = BOOLEAN_DEFAULT_COLUMNS
    if dict_type_cols is None:
        dict_type_cols = DICT_TYPE_COLUMNS
    
    # Add missing columns with appropriate defaults
    for col in required_columns:
        if col not in df.columns:
            if col in numeric_zero_cols:
                df[col] = 0
            elif col in NUMERIC_NULLABLE_COLUMNS:
                df[col] = pd.NA
            elif col in boolean_defaults:
                df[col] = boolean_defaults[col]
            else:
                df[col] = pd.NA

    for col in df.columns:
        if col in NUMERIC_ZERO_DEFAULT_COLUMNS:
            df[col] = df[col].fillna(0)
        elif col in NUMERIC_NULLABLE_COLUMNS:
            # Keep NaN as-is; only coerce non-numeric strings to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif col in BOOLEAN_DEFAULT_COLUMNS:
            df[col] = df[col].fillna(BOOLEAN_DEFAULT_COLUMNS[col]).astype(bool)
        elif col in DICT_TYPE_COLUMNS:
            df[col] = df[col].fillna({})
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if not pd.isna(x) else pd.NA)
            df[col] = df[col].astype(pd.StringDtype())
        else:
            df[col] = df[col].astype(pd.StringDtype())

    if {'freq_cardinality_ratio', 'freq_cardinality', 'vocab_size'}.issubset(df.columns):
        vocab_size = pd.to_numeric(df['vocab_size'], errors='coerce')
        computed_ratio = pd.to_numeric(df['freq_cardinality'], errors='coerce').divide(vocab_size.where(vocab_size > 0)).round(5)
        df['freq_cardinality_ratio'] = df['freq_cardinality_ratio'].combine_first(computed_ratio)

    # Drop any stale ebpb_rd column (legacy metric, no longer in schema).
    if 'ebpb_rd' in df.columns:
        df = df.drop(columns=['ebpb_rd'])

    if {'ebpb', 'total_tokens', 'total_bytes', 'byte_edit_distance_total', 'freq_cardinality'}.issubset(df.columns):
        total_tokens_s = pd.to_numeric(df['total_tokens'], errors='coerce')
        total_bytes_s = pd.to_numeric(df['total_bytes'], errors='coerce')
        byte_edit_distance = pd.to_numeric(df['byte_edit_distance_total'], errors='coerce')
        freq_cardinality = pd.to_numeric(df['freq_cardinality'], errors='coerce')
        observed_vocab = freq_cardinality.where(freq_cardinality > 0).clip(lower=2)
        computed_ebpb = (
            (total_tokens_s * np.log2(observed_vocab) + 8 * byte_edit_distance)
            .divide(total_bytes_s.where(total_bytes_s > 0))
            .clip(lower=0)
            .round(3)
        )
        # Overwrite any stale stored values with the new-formula result;
        # fall back to stored value only when raw totals are unavailable.
        df['ebpb'] = computed_ebpb.combine_first(pd.to_numeric(df['ebpb'], errors='coerce'))

    if {'ebpb_full', 'total_tokens', 'total_bytes', 'byte_edit_distance_total', 'vocab_size'}.issubset(df.columns):
        total_tokens_s = pd.to_numeric(df['total_tokens'], errors='coerce')
        total_bytes_s = pd.to_numeric(df['total_bytes'], errors='coerce')
        byte_edit_distance = pd.to_numeric(df['byte_edit_distance_total'], errors='coerce')
        vocab_size_s = pd.to_numeric(df['vocab_size'], errors='coerce').where(lambda x: x > 0).clip(lower=2)
        computed_ebpb_full = (
            (total_tokens_s * np.log2(vocab_size_s) + 8 * byte_edit_distance)
            .divide(total_bytes_s.where(total_bytes_s > 0))
            .clip(lower=0)
            .round(3)
        )
        df['ebpb_full'] = computed_ebpb_full.combine_first(pd.to_numeric(df['ebpb_full'], errors='coerce'))

    if {'freq_90pct_mass_ratio', 'freq_90pct_mass_count', 'vocab_size'}.issubset(df.columns):
        vocab_size = pd.to_numeric(df['vocab_size'], errors='coerce')
        computed_ratio = pd.to_numeric(df['freq_90pct_mass_count'], errors='coerce').divide(vocab_size.where(vocab_size > 0)).round(5)
        df['freq_90pct_mass_ratio'] = df['freq_90pct_mass_ratio'].combine_first(computed_ratio)
    
    # Reorder columns to match required_columns order where possible
    existing_cols = [col for col in required_columns if col in df.columns]
    other_cols = [col for col in df.columns if col not in required_columns]
    return df.reindex(columns=existing_cols + other_cols)