import pandas as pd
from transformers import AutoTokenizer
import random
import traceback
import json
from dataclasses import dataclass, field


@dataclass
class HighlightedText:
    """Lightweight replacement for gr.HighlightedText."""
    value: list = field(default_factory=list)
    color_map: dict = field(default_factory=dict)
    label: str = ""

# Moved from app.py
def decode_bpe_tokens(tokens):
    fixed_tokens = []
    for token in tokens:
        if token.startswith("Ġ"):
            try:
                fixed_token = " " + token[1:].encode("utf-8").decode("utf-8")
            except UnicodeDecodeError:
                fixed_token = token
        else:
            try:
                fixed_token = token.encode("utf-8").decode("utf-8")
            except UnicodeDecodeError:
                fixed_token = token
        fixed_tokens.append(fixed_token)
    return fixed_tokens

# Moved from app.py
def generate_distinct_colors(n):
    if n <= 0: # Handle edge case for no tokens
        return []
    if n > 256**3:
        raise ValueError("Cannot generate more than 16,777,216 unique colors.")
    spacing = int((256 * 256 * 256) ** (1 / 3) / n ** (1 / 3))
    spacing = max(1, spacing) # Ensure spacing is at least 1
    max_val = 256 - spacing
    used_colors = set()
    result = []
    attempts = 0
    max_attempts = n * 10 # Set a reasonable max attempts limit

    while len(result) < n and attempts < max_attempts:
        r = random.randint(0, max_val)
        g = random.randint(0, max_val)
        b = random.randint(0, max_val)
        r = min(255, r * spacing)
        g = min(255, g * spacing)
        b = min(255, b * spacing)
        color = f"#{r:02X}{g:02X}{b:02X}"
        if color not in used_colors:
            used_colors.add(color)
            result.append(color)
            attempts = 0 # Reset attempts on success
        else:
            attempts += 1
            if attempts % 50 == 0 and attempts > 0: # More aggressive spacing adjustment
                spacing = max(1, spacing - 1)
                max_val = 256 - spacing if spacing > 0 else 0 # Prevent max_val < 0
    
    if len(result) < n:
        # Fallback if distinct colors cannot be easily found (e.g., n is very large)
        # Fill remaining with random (possibly non-distinct) colors
        print(f"Warning: Could not generate {n} fully distinct colors. Generated {len(result)}. Filling rest randomly.")
        for _ in range(n - len(result)):
            r, g, b = [random.randint(0, 255) for _ in range(3)]
            result.append(f"#{r:02X}{g:02X}{b:02X}")

    return result

def format_model_display_text(model_name, revision="main", subfolder=None, _target_mode=False):
    """Format model display text similar to app.py format_model_link but without the URL."""
    display_text = model_name
    
    if revision != 'main':
        display_text += f" (rev: {revision})"
    if subfolder:
        display_text += f" (sub: {subfolder})"
    if _target_mode:
        display_text += f" (target)"
        
    return display_text

def parse_model_selection(model_selection):
    """Parse the model selection string to extract model_name, revision, subfolder, and target_mode."""
    if not model_selection:
        return None, "main", None, False
    
    # Try to parse as JSON first (for encoded model info)
    if model_selection.startswith('{'):
        try:
            model_info = json.loads(model_selection)
            return (
                model_info.get('model'),
                model_info.get('revision', 'main'),
                model_info.get('subfolder'),
                model_info.get('_target_mode', False)
            )
        except json.JSONDecodeError:
            pass
    
    # Fallback: assume it's just a model name
    return model_selection, "main", None, False

def create_model_choices(available_models_data):
    """Create formatted model choices for dropdown from model data."""
    choices = []
    
    for model_info in available_models_data:
        if isinstance(model_info, dict):
            model_name = model_info.get('model', '')
            revision = model_info.get('revision', 'main')
            subfolder = model_info.get('subfolder')
            _target_mode = model_info.get('_target_mode', False)
            
            # Create display text
            display_text = format_model_display_text(model_name, revision, subfolder, _target_mode)
            
            # Create value as JSON string for easy parsing
            value = json.dumps({
                'model': model_name,
                'revision': revision,
                'subfolder': subfolder,
                '_target_mode': _target_mode
            })
            
            choices.append((display_text, value))
        else:
            # Handle simple model name strings (fallback)
            choices.append((str(model_info), str(model_info)))
    
    return choices

# Moved from app.py
def _tokenize_single_model(text, chosen_model_selection, better_tokenization=False):
    try:
        # Parse the model selection
        model_name, revision, subfolder, _target_mode = parse_model_selection(chosen_model_selection)
        
        if not model_name:
            error_val = [("Error: No model selected", None)]
            return HighlightedText(value=error_val, label="Selection Error"), 0, "Error", pd.DataFrame()
        
        # Load tokenizer with the specified parameters
        kwargs = {
            'trust_remote_code': True,
            'revision': revision,
        }
        if subfolder:
            kwargs['subfolder'] = subfolder
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                **kwargs
            )
        except Exception as e:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                **kwargs
            )
        
        # Apply target mode if requested (for tokenizers like Marian with separate source/target vocabs)
        if _target_mode:
            if hasattr(tokenizer, '_switch_to_target_mode'):
                try:
                    tokenizer._switch_to_target_mode()
                except Exception as e:
                    error_val = [(f"Error switching to target mode: {e}", None)]
                    return HighlightedText(value=error_val, label="Target Mode Error"), 0, "Error", pd.DataFrame()
            elif hasattr(tokenizer, 'as_target_tokenizer'):
                try:
                    tokenizer.as_target_tokenizer()
                except Exception as e:
                    error_val = [(f"Error switching to target mode: {e}", None)]
                    return HighlightedText(value=error_val, label="Target Mode Error"), 0, "Error", pd.DataFrame()
        
        tokenizer.model_max_length = int(1e9)

        # Tokenize the text
        tokenized_text_raw = tokenizer.tokenize(text)
        
        num_tokens_raw = len(tokenized_text_raw)
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else "N/A"

        if num_tokens_raw == 0:
             return HighlightedText(), 0, vocab_size, pd.DataFrame()

        final_token_segments_for_display = []
        if better_tokenization:
            current_segments = decode_bpe_tokens(tokenized_text_raw)
            remaining_text = text 
            for token_match_segment in current_segments:
                found_match = False
                for i in range(1, len(remaining_text) + 1):
                    segment = remaining_text[:i]
                    temp_tokenized_segment_decoded = decode_bpe_tokens(tokenizer.tokenize(segment))
                    if temp_tokenized_segment_decoded and temp_tokenized_segment_decoded[0] == token_match_segment:
                         final_token_segments_for_display.append(segment)
                         remaining_text = remaining_text[len(segment):]
                         found_match = True
                         break
                if not found_match:
                    final_token_segments_for_display.append(token_match_segment)
                    if remaining_text.startswith(token_match_segment):
                        remaining_text = remaining_text[len(token_match_segment):]
        else:
            final_token_segments_for_display = decode_bpe_tokens(tokenized_text_raw)

        num_display_segments = len(final_token_segments_for_display)
        if num_display_segments == 0:
            return HighlightedText(), 0, vocab_size, pd.DataFrame()
            
        random_colors = generate_distinct_colors(num_display_segments)

        output_highlight = []
        color_map = {}
        for idx, token_display_text in enumerate(final_token_segments_for_display):
             token_label = str(idx)
             output_highlight.append((token_display_text, token_label))
             color_map[token_label] = random_colors[idx % len(random_colors)] 
        
        original_tokens_with_ids = tokenizer(text, add_special_tokens=False, truncation=False)
            
        original_token_texts_for_table = tokenizer.convert_ids_to_tokens(original_tokens_with_ids['input_ids'])
        processed_tokens_for_table = decode_bpe_tokens(original_token_texts_for_table)
        
        table_data = []
        for i, token_id in enumerate(original_tokens_with_ids['input_ids']):
            original_token = original_token_texts_for_table[i]
            display_token_text = processed_tokens_for_table[i]
            try:
                 utf8_bytes_repr = repr(display_token_text.encode('utf-8'))
            except Exception:
                 utf8_bytes_repr = "Error encoding"
            table_data.append({
                "TokenID": token_id, "Token": original_token,
                "Text": display_token_text, "UTF8 Bytes": utf8_bytes_repr
            })
        token_df = pd.DataFrame(table_data)
        
        token_count = len(original_tokens_with_ids['input_ids'])
        
        return HighlightedText(value=output_highlight, color_map=color_map, label="Tokenized Output"), token_count, vocab_size, token_df

    except Exception as e:
        print(f"Error tokenizing text with model selection {chosen_model_selection}: {e}")
        traceback.print_exc()
        error_val = [(f"Error: {e}", None)]
        return HighlightedText(value=error_val, label="Tokenization Error"), 0, "Error", pd.DataFrame()

# Moved from app.py
def tokenize_text(text, chosen_model_selection_1, chosen_model_selection_2, better_tokenization=False):
    if not text:
        empty_highlight = HighlightedText()
        empty_df = pd.DataFrame()
        return empty_highlight, 0, "N/A", empty_df, empty_highlight, 0, "N/A", empty_df

    if not chosen_model_selection_1 or not chosen_model_selection_2:
        empty_highlight = HighlightedText()
        empty_df = pd.DataFrame()
        return empty_highlight, 0, "N/A", empty_df, empty_highlight, 0, "N/A", empty_df

    output1, count1, vocab1, table1 = _tokenize_single_model(text, chosen_model_selection_1, better_tokenization)
    output2, count2, vocab2, table2 = _tokenize_single_model(text, chosen_model_selection_2, better_tokenization)
    return output1, count1, vocab1, table1, output2, count2, vocab2, table2
