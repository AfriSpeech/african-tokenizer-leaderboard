"""Export the leaderboard data for the static Space: one JSON file the browser ranks from.

Per tokenizer and language (all of the language's text together):
  fertility        tokens / words
  bytes_per_token  bytes / tokens
  parity           tokens on the language's parity verses / tokens on the same verses in
                   English, same tokenizer (parity_en__<lang> sets; null where none)

Usage:
  python benchmark/export_static.py RESULTS_DIR DATASET_META_YAML frontend/public/data/leaderboard.json
"""

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

OVERALL = {"lang_overall", "multilingual_overall"}

# Friendly names; everything else shows its repo name and owner.
DISPLAY = {
    "Xenova/gpt-4o": ("GPT-4o (o200k)", "OpenAI"),
    "Xenova/gpt-4": ("GPT-4 (cl100k)", "OpenAI"),
    "openai/gpt-oss-20b": ("gpt-oss", "OpenAI"),
    "tokenizers/pocket-tts-english": ("Pocket TTS English", "Kyutai"),
    "tokenizers/pocket-tts-en-twi": ("Pocket TTS English+Twi", "GhanaOpenAI"),
    "tokenizers/serengeti": ("serengeti", "UBC-NLP"),
}
# Where the browser loads each tokenizer for "Compare": a Hub repo id, "local:<folder>" for
# folders shipped with the Space, or None when it can't run in a browser (gated repos).
BROWSER = {
    "tokenizers/pocket-tts-english": "local:pocket-tts-english",
    "tokenizers/pocket-tts-en-twi": "local:pocket-tts-en-twi",
    "tokenizers/serengeti": "local:serengeti",
    # Repos without a tokenizer.json, converted with transformers (identical counts verified).
    "castorini/afriberta_large": "local:afriberta",
    "facebook/seamless-m4t-v2-large": "local:seamless-m4t-v2",
    "google/mt5-base": "local:mt5",
    "google/byt5-small": "builtin:bytes",  # one token per UTF-8 byte
    "facebook/m2m100_418M": None,  # slow-only tokenizer, no tokenizer.json conversion
    "meta-llama/Llama-3.1-8B-Instruct": None,
    "google/gemma-3-27b-it": None,
    "google/gemma-2-9b": None,
    "CohereLabs/aya-expanse-8b": None,
    "lelapa/InkubaLM-0.4B": None,
}
# Architecture of the model each tokenizer belongs to (filter on the leaderboard).
MODEL_TYPE = {
    **{m: "decoder" for m in [
        "Xenova/gpt-4o", "Xenova/gpt-4", "openai/gpt-oss-20b", "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-3-27b-it", "google/gemma-2-9b", "Qwen/Qwen3-8B",
        "mistralai/Mistral-Nemo-Instruct-2407", "mistralai/Mistral-7B-Instruct-v0.3",
        "deepseek-ai/DeepSeek-V3", "microsoft/phi-4", "CohereLabs/aya-expanse-8b",
        "bigscience/bloom", "lelapa/InkubaLM-0.4B"]},
    **{m: "encoder" for m in [
        "FacebookAI/xlm-roberta-base", "google-bert/bert-base-multilingual-cased",
        "Davlan/afro-xlmr-large", "castorini/afriberta_large", "tokenizers/serengeti"]},
    **{m: "encoder-decoder" for m in [
        "google/mt5-base", "google/byt5-small", "google/madlad400-3b-mt",
        "facebook/nllb-200-distilled-600M", "facebook/m2m100_418M", "CohereLabs/aya-101"]},
    **{m: "speech" for m in [
        "openai/whisper-large-v3", "facebook/seamless-m4t-v2-large",
        "tokenizers/pocket-tts-english", "tokenizers/pocket-tts-en-twi"]},
}

# Source repo for the "view on Hugging Face" link.
SOURCE = {
    "tokenizers/pocket-tts-english": "kyutai/pocket-tts-without-voice-cloning",
    "tokenizers/pocket-tts-en-twi": "https://github.com/GhanaOpenAI/pocket-tts-twi",
    "tokenizers/serengeti": "UBC-NLP/serengeti",
}


def main(results_dir: str, meta_yaml: str, out: str) -> None:
    meta = yaml.safe_load(open(meta_yaml, encoding="utf-8"))
    companions = {r["code"]: r["companion"] for r in csv.DictReader(
        open(Path(__file__).parent / "companion_languages.csv", encoding="utf-8"))}
    languages = sorted(
        ({"code": c, "name": m["pretty_name"], "group": m.get("group", "african"),
          "companion": companions.get(c), "has_parity": bool(m.get("parities"))}
         for c, m in meta.items()),
        key=lambda x: (x["group"] != "african", x["name"].lower()))
    index = {x["code"]: i for i, x in enumerate(languages)}

    tokenizers = []
    for f in sorted(glob.glob(f"{results_dir}/evaluations/results_*.jsonl")):
        rows = [json.loads(line) for line in open(f, encoding="utf-8")]
        model = rows[0]["model"]
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        par_t: dict[str, float] = {}
        par_r: dict[str, float] = {}
        for r in rows:
            if r["evaluation_type"] == "standard" and r["domain"] not in OVERALL:
                s = sums[r["lang"]]
                s[0] += r["total_tokens"]; s[1] += r["total_words"]; s[2] += r["total_bytes"]
            elif r["evaluation_type"] == "parity":
                (par_r if r.get("reference") else par_t)[r["domain"]] = r["total_tokens"]
        n = len(languages)
        fert, bpt, par = [None] * n, [None] * n, [None] * n
        for lang, (t, w, b) in sums.items():
            if lang in index and t and w:
                fert[index[lang]] = round(t / w, 4)
                bpt[index[lang]] = round(b / t, 3)
        for dom, t in par_t.items():
            lang = dom.split("__", 1)[1] if "__" in dom else None
            if lang in index and par_r.get(dom):
                par[index[lang]] = round(t / par_r[dom], 4)
        missing = [languages[i]["code"] for i in range(n) if fert[i] is None]
        name, org = DISPLAY.get(model, (model.rpartition("/")[2], model.rpartition("/")[0]))
        info = rows[0]
        tokenizers.append({
            "model": model, "name": name, "org": org,
            "vocab_size": info.get("vocab_size"),
            "algorithm": info.get("tokenizer_algorithm"),
            "type": MODEL_TYPE.get(model, "decoder"),
            "browser": BROWSER.get(model, model),
            "source": SOURCE.get(model, model),
            "fertility": fert, "bytes_per_token": bpt, "parity": par,
        })
        if model not in MODEL_TYPE:
            print(f"  NOTE: {model} has no MODEL_TYPE entry; defaulting to decoder")
        print(f"{model}: {n - len(missing)}/{n} languages" + (f", missing {missing}" if missing else ""))

    data = {"languages": languages, "tokenizers": tokenizers}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {out}: {len(tokenizers)} tokenizers x {len(languages)} languages, "
          f"{Path(out).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
