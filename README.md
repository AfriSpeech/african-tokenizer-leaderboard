---
title: African Tokenizer Leaderboard
emoji: 🌍
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: How efficiently tokenizers handle 313 African languages
tags:
- leaderboard
- tokenizers
- african-languages
- multilingual
---

# African Tokenizer Leaderboard

How efficiently tokenizers handle African languages: tokens per word, parity vs English and
bytes per token for 313 African languages (plus English, French, Arabic, Portuguese and
Spanish), on text from the AfriSpeech Africa Corpus. See `docs/about.md` for the metrics and
the benchmark.

## Layout

- `benchmark/` — builds the benchmark (`build_african_benchmark.py`), the tokenizer lineup
  (`tokenizers.txt`) and the parallel evaluation runner (`run_evals.sh`).
- `tokenizer_evaluate.py`, `results_aggregator.py` — evaluation and aggregation (from the
  original leaderboard; parity pairs each language with its own English reference).
- `api.py` + `api_v2.py` — FastAPI backend; `frontend/` — React UI.
- `tokenizers/` — Pocket TTS tokenizers wrapped for `AutoTokenizer`.

## Data

The benchmark text (`AfriSpeech/african-tokenizer-benchmark`) is private: most of it is
publisher-copyrighted. Results go to `AfriSpeech/african-tokenizer-results`. The Space needs an
`HF_TOKEN` secret with read access to both (write access to results for submissions).

## Credits

Based on the [Multilingual Tokenizer Leaderboard](https://huggingface.co/spaces/eduagarcia/multilingual-tokenizer-leaderboard)
by Eduardo A. S. Garcia (Apache-2.0); see `NOTICE`.
