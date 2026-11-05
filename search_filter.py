import re
import textwrap
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional

class SearchFilter:
    """Advanced search filter for tokenizer evaluation data with support for text search and numeric operators."""
    
    def __init__(self):
        """Initialize the search filter with supported operators and numeric fields."""
        self.operators = {
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '=': lambda x, y: abs(x - y) < 1e-10,  # Handle floating point equality
            '==': lambda x, y: abs(x - y) < 1e-10,
            '!=': lambda x, y: abs(x - y) >= 1e-10,
        }
        
        # Define numeric fields that can be filtered
        self.numeric_fields = {
            'fertility', 'compression_chars', 'compression_bytes', 'reversible_ratio',
            'unk_ratio', 'total_tokens', 'total_words', 'total_chars', 'total_bytes',
            'byte_edit_distance_total', 'char_edit_distance_total',
            'total_docs', 'reversible_docs_count', 'unk_tokens_count', 'vocab_size',
            'vocab_near_duplicates', 'parity_en',
            # Experimental fidelity metrics
            'byte_fidelity', 'effective_bytes_per_token', 'ebpb', 'ebpb_full',
            'char_fidelity', 'effective_chars_per_token', 'ebpc',
            # Experimental frequency / Zipf metrics
            'freq_cardinality', 'freq_cardinality_ratio', 'freq_auc', 'freq_slope', 'freq_power_law',
            'freq_entropy', 'freq_renyi_entropy',
            'freq_shannon_efficiency', 'freq_renyi_efficiency',
            'freq_shannon_efficiency_full', 'freq_renyi_efficiency_full',
            'freq_percentile_freq',
            'freq_hapax_count', 'freq_hapax_ratio',
            'freq_half_mass_count', 'freq_90pct_mass_count', 'freq_90pct_mass_ratio', 'freq_gini',
        }
        
        # Define percentage fields (values are stored as decimals but users might input as percentages)
        self.percentage_fields = {
            'reversible_ratio', 'unk_ratio', 'vocab_near_duplicates',
            'byte_fidelity', 'char_fidelity',
            'freq_hapax_ratio', 'freq_gini',
            'freq_shannon_efficiency', 'freq_renyi_efficiency',
            'freq_shannon_efficiency_full', 'freq_renyi_efficiency_full',
            'freq_percentile_freq', 'freq_cardinality_ratio', 'freq_90pct_mass_ratio',
        }
        
        # Define text search fields
        self.text_fields = {
            'model', 'revision', 'subfolder', '_target_mode', 'tokenizer_algorithm', 'tokenizer_implementation', 'vocab_hash', 'tokenizer_config_hash'
        }
    
    def parse_search_query(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Parse a search query into text search and field filters.
        
        Args:
            query: Search query like "bert @fertility>1 @unk_ratio<0.01 @reversible_ratio=1"
            
        Returns:
            Tuple of (text_search, field_filters)
        """
        if not query or not query.strip():
            return "", []
            
        # Split the query into parts
        parts = query.strip().split()
        
        text_parts = []
        field_filters = []
        
        for part in parts:
            if part.startswith('@'):
                # This is a field filter
                field_filter = self._parse_field_filter(part[1:])  # Remove @ prefix
                if field_filter:
                    field_filters.append(field_filter)
            else:
                # This is part of text search
                text_parts.append(part)
        
        text_search = " ".join(text_parts)
        return text_search, field_filters
    
    def _parse_field_filter(self, filter_str: str) -> Optional[Dict]:
        """
        Parse a field filter like 'fertility>1' or 'unk_ratio<0.01'.
        
        Args:
            filter_str: Field filter string
            
        Returns:
            Dictionary with field, operator, and value, or None if invalid
        """
        # Match field and operator-value pairs
        # Support operators: >=, <=, >, <, =, ==, !=
        pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)(>=|<=|>|<|==|!=|=)(.+)$'
        match = re.match(pattern, filter_str)
        
        if not match:
            return None
            
        field, operator, value_str = match.groups()
        
        # Check if field is valid
        if field not in self.numeric_fields:
            return None
            
        # Convert value to float
        try:
            value = float(value_str)
            
            # Handle percentage fields - if user inputs value > 1, assume it's a percentage
            if field in self.percentage_fields and value > 1:
                value = value / 100.0
                
        except ValueError:
            return None
            
        return {
            'field': field,
            'operator': operator,
            'value': value
        }
    
    def apply_filters(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Apply search filters to a DataFrame.
        
        Args:
            df: Input DataFrame
            query: Search query string
            
        Returns:
            Filtered DataFrame
        """
        if df.empty or not query or not query.strip():
            return df
            
        # Parse the query
        text_search, field_filters = self.parse_search_query(query)
        
        # Start with the original DataFrame
        filtered_df = df.copy()
        
        # Apply text search
        if text_search:
            filtered_df = self._apply_text_search(filtered_df, text_search)
        
        # Apply field filters
        for field_filter in field_filters:
            filtered_df = self._apply_field_filter(filtered_df, field_filter)
        
        return filtered_df
    
    def _apply_text_search(self, df: pd.DataFrame, text_search: str) -> pd.DataFrame:
        """Apply text search to specified fields."""
        if df.empty:
            return df
            
        # Create a mask for text search
        mask = pd.Series(False, index=df.index)
        
        for field in self.text_fields:
            if field in df.columns:
                # Case-insensitive search
                temp_field = df[field]
                if field == 'revision':
                    temp_field = temp_field.apply(lambda x: f"(rev: {x})" if x != "main" else "").astype(str)
                elif field == 'subfolder':
                    temp_field = temp_field.apply(lambda x: f"(sub: {x})" if pd.notna(x) else "").astype(str)
                elif field == '_target_mode':
                    temp_field = temp_field.apply(lambda x: f"(target)" if x else "").astype(str)
                field_mask = temp_field.str.contains(text_search, case=False, na=False)
                mask = mask | field_mask
        
        return df[mask]
    
    def _apply_field_filter(self, df: pd.DataFrame, field_filter: Dict) -> pd.DataFrame:
        """Apply a single field filter to the DataFrame."""
        if df.empty:
            return df
            
        field = field_filter['field']
        operator = field_filter['operator']
        value = field_filter['value']
        
        if field not in df.columns:
            return df
            
        # Get the operator function
        op_func = self.operators.get(operator)
        if not op_func:
            return df
            
        # Apply the filter
        try:
            mask = df[field].apply(lambda x: op_func(float(x), value) if pd.notna(x) else False)
            return df[mask]
        except (ValueError, TypeError):
            # If there's an error in comparison, return original df
            return df
    
    def get_filter_help(self) -> str:
        """Get help text for the filter syntax."""
        return textwrap.dedent("""
        ## How to Search & Filter

        ### Text Search
        Type any word to find models by name, algorithm, or hash.
        Multiple words are matched together (AND).

        | Query | What it finds |
        |-------|--------------|
        | `bert` | All models with "bert" in the name |
        | `gpt tokenizer` | Models matching both "gpt" and "tokenizer" |
        | `BPE` | Models using the BPE algorithm |

        ### Field Filters
        Use `@field_name` followed by an operator and value to filter by numeric columns.
        No spaces between the field, operator, and value.

        **Syntax:** `@field` `operator` `value`

        **Operators:**

        | Operator | Meaning |
        |----------|---------|
        | `>` | Greater than |
        | `<` | Less than |
        | `>=` | Greater than or equal |
        | `<=` | Less than or equal |
        | `=` or `==` | Equal to |
        | `!=` | Not equal to |

        **Examples:**

        | Filter | Meaning |
        |--------|---------|
        | `@fertility<2` | Fertility below 2 tokens/word |
        | `@unk_ratio<0.01` | Less than 1% unknown tokens |
        | `@reversible_ratio>0.95` | Over 95% reversible |
        | `@vocab_size>=50000` | At least 50k vocabulary |
        | `@tokenizer_algorithm=BPE` | Only BPE tokenizers |

        > **Tip — percentage fields:** `reversible_ratio`, `unk_ratio`, `vocab_near_duplicates`, `byte_fidelity`, `char_fidelity`, and frequency efficiency fields are stored as decimals (0–1).
        > You can write either `@unk_ratio<0.01` or `@unk_ratio<1` — values above 1 are automatically treated as percentages.

        ### Available Fields

        | Field | Description | Type |
        |-------|-------------|------|
        | `fertility` | Tokens per word (lower = better) | Numeric |
        | `compression_bytes` | Bytes per token | Numeric |
        | `reversible_ratio` | Fraction of reversible documents (0–1) | Percentage |
        | `unk_ratio` | Fraction of unknown tokens (0–1) | Percentage |
        | `parity_en` | Token ratio vs English (1.0 = parity) | Numeric |
        | `vocab_size` | Vocabulary size | Numeric |
        | `vocab_near_duplicates` | Near-duplicate vocab entries (0–1) | Percentage |
        | `total_tokens` | Total tokens in evaluation corpus | Numeric |
        | `total_bytes` | Total UTF-8 bytes in evaluation corpus | Numeric |
        | `total_chars` | Total Unicode characters in evaluation corpus | Numeric |
        | `byte_edit_distance_total` | Total byte-level Levenshtein edit distance (lower is better) | Numeric |
        | `char_edit_distance_total` | Total char-level Levenshtein edit distance (lower is better) | Numeric |

        ### Experimental Fields

        **Fidelity metrics** (continuous replacement for `reversible_ratio`):

        | Field | Description | Type |
        |-------|-------------|------|
        | `byte_fidelity` | Byte-level reconstruction fidelity (1.0 = perfect, higher is better) | Percentage |
        | `effective_bytes_per_token` | Compression adjusted for byte fidelity (higher is better) | Numeric |
        | `ebpb` | Effective bits per byte — observed-cardinality indexing plus raw-byte fallback cost (lower is better) | Numeric |
        | `ebpb_full` | Effective bits per byte using full declared vocabulary size (lower is better) | Numeric |
        | `char_fidelity` | Character-level reconstruction fidelity (1.0 = perfect, higher is better) | Percentage |
        | `effective_chars_per_token` | Compression adjusted for char fidelity (higher is better) | Numeric |
        | `ebpc` | Effective bits per char — log2(vocab) / eff_chars_per_token (lower is better) | Numeric |

        **Frequency / Zipf distribution metrics** (Zouhar et al., ACL 2023; Lotz et al., 2025):

        | Field | Description | Type |
        |-------|-------------|------|
        | `freq_cardinality` | Unique token types observed in corpus | Numeric |
        | `freq_cardinality_ratio` | Unique token types observed as a share of vocabulary size | Percentage |
        | `freq_auc` | Area under log-rank/log-freq curve (higher = richer usage) | Numeric |
        | `freq_slope` | Zipf slope of rank-freq curve (more negative = steeper drop-off) | Numeric |
        | `freq_power_law` | Deviation from Zipf power-law fit (MAE, lower = purer power law) | Numeric |
        | `freq_entropy` | Shannon entropy of token frequencies in bits (higher = more uniform) | Numeric |
        | `freq_renyi_entropy` | Rényi entropy α=3 in bits (higher = better) | Numeric |
        | `freq_shannon_efficiency` | Shannon entropy / log2(observed vocab) — 0–1 (higher = better) | Percentage |
        | `freq_renyi_efficiency` | **Rényi efficiency α=3 / log2(obs vocab) — strongest MT predictor (higher = better)** | Percentage |
        | `freq_shannon_efficiency_full` | Shannon efficiency normalized by full vocab size | Percentage |
        | `freq_renyi_efficiency_full` | Rényi efficiency normalized by full vocab size | Percentage |
        | `freq_percentile_freq` | Mid-frequency mass P3–P83 — 2nd-best MT predictor (higher = better) | Percentage |
        | `freq_hapax_count` | Number of once-only token types | Numeric |
        | `freq_hapax_ratio` | Fraction of once-only token types (lower = better) | Percentage |
        | `freq_half_mass_count` | Types needed to cover 50% of corpus (lower = more concentrated) | Numeric |
        | `freq_90pct_mass_count` | Types needed to cover 90% of corpus (higher = more uniform) | Numeric |
        | `freq_90pct_mass_ratio` | Types needed to cover 90% of corpus as a share of vocabulary size | Percentage |
        | `freq_gini` | Gini inequality of frequencies (0 = uniform, 1 = single dominant, lower = better) | Percentage |

        ### Combining Text + Filters

        Mix free text and `@field` filters in a single query. All conditions must match (AND).

        ```
        bert @fertility<2 @unk_ratio<0.01
        ```
        Finds models with "bert" in the name, fertility below 2, and less than 1% unknown tokens.

        ```
        @vocab_size<100000 @unk_ratio<0.05 @reversible_ratio>0.25
        ```
        Vocabulary under 100k, unknown ratio under 5%, and at least 25% reversible.

        ```
        @ebpb<2.5 @byte_fidelity>0.99 @freq_renyi_efficiency>0.55
        ```
        Low effective bits per byte, near-perfect byte reconstruction, and high Rényi efficiency.
        """).strip()
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate a search query and return any error messages.
        
        Args:
            query: Search query to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return True, ""
            
        try:
            text_search, field_filters = self.parse_search_query(query)
            
            # Check each field filter
            for field_filter in field_filters:
                field = field_filter['field']
                operator = field_filter['operator']
                
                if field not in self.numeric_fields:
                    return False, f"Unknown field: {field}"
                    
                if operator not in self.operators:
                    return False, f"Unknown operator: {operator}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Parse error: {str(e)}"

# Create a global instance
search_filter = SearchFilter() 