import pandas as pd
import yaml
from pathlib import Path
from tokenizer_evaluate import download_repos_once
from results_aggregator import load_all_model_results

def update_readme_models():
    """
    Downloads the latest tokenizer evaluation results,
    extracts all unique model names, and updates the
    models list in the README.md YAML frontmatter.
    """
    print("Downloading benchmark and results repositories...")
    _, results_dir, _, _ = download_repos_once()

    print("Loading all model results...")
    all_results_df = load_all_model_results(results_dir)

    if all_results_df.empty:
        print("No results found. README.md will not be updated.")
        return

    model_names = sorted(all_results_df['model'].unique())
    print(f"Found {len(model_names)} unique models.")

    readme_path = Path('README.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find YAML frontmatter
    parts = content.split('---')
    if len(parts) < 3:
        print("Error: README.md does not have a valid YAML frontmatter.")
        return

    frontmatter_str = parts[1]
    body = '---'.join(parts[2:])

    try:
        # Using RoundTripLoader to preserve comments and formatting if available
        data = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML from README.md: {e}")
        return

    data['models'] = model_names

    new_frontmatter_str = yaml.dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)

    new_content = f"---\n{new_frontmatter_str.strip()}\n---\n{body.lstrip()}"

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("README.md has been updated with the latest model list.")

if __name__ == "__main__":
    update_readme_models() 