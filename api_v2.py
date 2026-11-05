"""Focused endpoints for the African leaderboard UI, computed from the raw result rows.

Per tokenizer and language:
  fertility        tokens / words              (lower is better)
  bytes_per_token  bytes / tokens              (higher is better)
  unk_ratio        unknown tokens / tokens
  parity           tokens on the language's parity verses / tokens on the same verses in
                   English, both with the same tokenizer (parity_en__<lang> sets)
"all" (every language) and "african" (African languages only) average per-language values
with equal weight per language, as does a custom set such as "twi,en".
"""

from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v2")

OVERALL_DOMAINS = {"lang_overall", "multilingual_overall"}
ALL_LANGS = "all"  # every language in the benchmark
ALL_AFRICAN = "african"  # the African languages only

# Friendly names for well-known tokenizers; everything else shows its repo name.
DISPLAY = {
    "Xenova/gpt-4o": ("GPT-4o (o200k)", "OpenAI"),
    "Xenova/gpt-4": ("GPT-4 (cl100k)", "OpenAI"),
    "openai/gpt-oss-20b": ("gpt-oss", "OpenAI"),
    "tokenizers/pocket-tts-english": ("Pocket TTS English", "Kyutai"),
    "tokenizers/pocket-tts-en-twi": ("Pocket TTS English+Twi", "GhanaOpenAI"),
}

_state: dict = {"meta": None, "raw": None, "version": None}


def configure(dataset_meta: dict, raw_results_df: pd.DataFrame, data_version: str) -> None:
    """Called by api.py once results are loaded (and again after a refresh)."""
    _state.update(meta=dataset_meta, raw=raw_results_df, version=data_version)
    _tables.cache_clear()


def _display(model: str) -> tuple[str, str]:
    if model in DISPLAY:
        return DISPLAY[model]
    org, _, name = model.rpartition("/")
    return name or model, org


def _languages() -> list[dict]:
    meta = _state["meta"] or {}
    out = []
    for code, m in meta.items():
        if not isinstance(m, dict) or "pretty_name" not in m:
            continue
        out.append({
            "code": code,
            "name": m["pretty_name"],
            "group": m.get("group", "african"),
            "has_parity": bool(m.get("parities")),
        })
    return sorted(out, key=lambda x: (x["group"] != "african", x["name"].lower()))


@lru_cache(maxsize=1)
def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per model/lang/domain sums, per model/lang parity, per model info)."""
    raw = _state["raw"]
    if raw is None or raw.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    raw = raw.copy()
    raw["reference"] = raw["reference"].fillna(False).astype(bool)
    std = raw[(raw["evaluation_type"] == "standard") & ~raw["domain"].isin(OVERALL_DOMAINS)]
    sums = (std.groupby(["model", "lang", "domain"], as_index=False)
            [["total_tokens", "total_words", "total_bytes", "unk_tokens_count"]].sum())

    par = raw[raw["evaluation_type"] == "parity"]
    tgt = par[~par["reference"]][["model", "lang", "domain", "total_tokens"]]
    ref = (par[par["reference"]][["model", "domain", "total_tokens"]]
           .rename(columns={"total_tokens": "ref_tokens"}))
    parity = tgt.merge(ref, on=["model", "domain"], how="left")
    parity["parity"] = parity["total_tokens"] / parity["ref_tokens"]
    parity = parity.groupby(["model", "lang"], as_index=False)["parity"].first()

    info_cols = [c for c in ("vocab_size", "tokenizer_algorithm", "tokenizer_class")
                 if c in raw.columns]
    info = raw.groupby("model", as_index=False)[info_cols].first()
    return sums, parity, info


def _per_language() -> pd.DataFrame:
    """Per tokenizer and language, over all of the language's text."""
    sums, parity, _ = _tables()
    if sums.empty:
        return sums
    g = sums.groupby(["model", "lang"], as_index=False)[
        ["total_tokens", "total_words", "total_bytes", "unk_tokens_count"]].sum()
    g["fertility"] = g["total_tokens"] / g["total_words"]
    g["bytes_per_token"] = g["total_bytes"] / g["total_tokens"]
    g["unk_ratio"] = g["unk_tokens_count"] / g["total_tokens"]
    return g.merge(parity, on=["model", "lang"], how="left")


def _clean(v: object, nd: int) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), nd)


@router.get("/languages")
def languages() -> dict:
    langs = _languages()
    return {
        "languages": langs,
        "n_african": sum(1 for x in langs if x["group"] == "african"),
        "data_version": _state["version"],
    }


def parse_langs(lang: str) -> list[str] | None:
    """'all' / 'african' -> None (a preset); 'twi,en' -> ['twi', 'en']."""
    if lang in (ALL_LANGS, ALL_AFRICAN):
        return None
    codes = list(dict.fromkeys(c.strip() for c in lang.split(",") if c.strip()))
    known = {x["code"] for x in _languages()}
    unknown = [c for c in codes if c not in known]
    if not codes or unknown:
        raise HTTPException(404, f"unknown language(s): {unknown or lang!r}")
    return codes


@router.get("/ranking")
def ranking(lang: str = ALL_LANGS) -> dict:
    """Rank tokenizers on one language, a custom set ("twi,en"), every language ("all") or
    the African languages ("african").
    Several languages: each counts equally (mean of per-language values); per_lang holds
    each selected language's tokens per word."""
    per = _per_language()
    african = {x["code"] for x in _languages() if x["group"] == "african"}
    codes = parse_langs(lang)
    if per.empty:
        return {"rows": [], "lang": lang, "n_african": len(african)}
    every = {x["code"] for x in _languages()}
    pool = set(codes) if codes is not None else (african if lang == ALL_AFRICAN else every)
    sel = per[per["lang"].isin(pool)]
    agg = sel.groupby("model", as_index=False).agg(
        fertility=("fertility", "mean"), bytes_per_token=("bytes_per_token", "mean"),
        unk_ratio=("unk_ratio", "mean"), parity=("parity", "mean"),
        n_langs=("lang", "nunique"))
    per_lang: dict[str, dict[str, float | None]] = {}
    if codes is not None and len(codes) > 1:
        for r in sel.itertuples(index=False):
            per_lang.setdefault(r.model, {})[r.lang] = _clean(r.fertility, 3)
    _, _, info = _tables()
    agg = agg.merge(info, on="model", how="left").sort_values("fertility")
    rows = []
    for rank, r in enumerate(agg.itertuples(index=False), 1):
        name, org = _display(r.model)
        row = {
            "rank": rank, "model": r.model, "name": name, "org": org,
            "algorithm": getattr(r, "tokenizer_algorithm", None),
            "vocab_size": None if pd.isna(getattr(r, "vocab_size", None))
            else int(r.vocab_size),
            "fertility": _clean(r.fertility, 3), "parity": _clean(r.parity, 3),
            "bytes_per_token": _clean(r.bytes_per_token, 2),
            "unk_ratio": _clean(r.unk_ratio, 4), "n_langs": int(r.n_langs),
        }
        if per_lang:
            row["per_lang"] = {c: per_lang.get(r.model, {}).get(c) for c in codes or []}
        rows.append(row)
    return {"rows": rows, "lang": lang, "langs": codes, "n_african": len(african),
            "n_langs": len(pool)}


@router.get("/tokenizer")
def tokenizer(name: str, lang: str = ALL_LANGS) -> dict:
    sums, parity, info = _tables()
    if sums.empty or name not in set(sums["model"]):
        raise HTTPException(404, f"unknown tokenizer {name!r}")
    langs = {x["code"]: x for x in _languages()}
    per_all = _per_language()
    mine = per_all[per_all["model"] == name]
    african = mine[mine["lang"].map(lambda c: langs.get(c, {}).get("group") == "african")]
    ranked = african.sort_values("fertility")

    def lang_rows(df: pd.DataFrame) -> list[dict]:
        return [{"code": r.lang, "name": langs[r.lang]["name"],
                 "fertility": _clean(r.fertility, 3), "parity": _clean(r.parity, 3)}
                for r in df.itertuples(index=False)]

    codes = parse_langs(lang)
    if codes is not None:
        pool = mine[mine["lang"].isin(set(codes))]
    elif lang == ALL_AFRICAN:
        pool = african
    else:
        pool = mine
    disp, org = _display(name)
    row = info[info["model"] == name].iloc[0]
    return {
        "model": name, "name": disp, "org": org,
        "vocab_size": None if pd.isna(row.get("vocab_size")) else int(row["vocab_size"]),
        "algorithm": row.get("tokenizer_algorithm"),
        "fertility": _clean(pool["fertility"].mean(), 3),
        "parity": _clean(pool["parity"].mean(), 3),
        "bytes_per_token": _clean(pool["bytes_per_token"].mean(), 2),
        "n_langs": int(pool["lang"].nunique()),
        "best_languages": lang_rows(ranked.head(5)),
        "worst_languages": lang_rows(ranked.tail(5).iloc[::-1]),
    }
