"""Build the African tokenizer benchmark in the layout tokenizer_evaluate.py reads.

Sources (via the Africa Corpus Builder conventions, AfriSpeech on Hugging Face):
  bible        AfriSpeech/africa-corpus         one translation per language, chapters as docs
  watchtower   AfriSpeech/african-corpus-jw     articles as docs
  web          AfriSpeech/madlad-91-african     clean split (@1), streamed; docs as-is
  web (Twi)    ghanaopenai/pristine-twi         Twi only
  parity       Bible verses aligned with English (one text per verse on each side)

Languages: every africa-corpus language with >= --min-sentences distinct sentences.
Each domain is sampled (seeded) up to --words-per-domain words; parity up to
--parity-verses verses. Output:

  OUT/<lang>/<domain>.jsonl.gz      {"text": ...} per document
  OUT/<lang>/parity.jsonl.gz        target-language verses
  OUT/en/parity_<lang>.jsonl.gz     the same verses in English, same order
  OUT/dataset_meta.yaml             what tokenizer_evaluate.py loads
  OUT/languages.csv                 code, name, sentences, domains (for the UI)

Usage:
  python benchmark/build_african_benchmark.py OUT [--langs twi ewe ...] [--min-sentences 10000]
"""

import argparse
import csv
import gzip
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from huggingface_hub import HfFileSystem, hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import africa_corpus as ac  # noqa: E402  (vendored from africa-corpus-builder)

csv.field_size_limit(sys.maxsize)

BIBLE_REPO = "AfriSpeech/africa-corpus"
JW_REPO = "AfriSpeech/african-corpus-jw"
WEB_REPO = "AfriSpeech/madlad-91-african"
PRISTINE_TWI = ("ghanaopenai/pristine-twi", "9aca5772a8bec63aafb3008dc2fc0df86227cb19")

# Two-letter / alternative codes used by the JW and MADLAD repos -> africa-corpus codes.
CODE_MAP = {
    "af": "afr", "ak": "aka", "am": "amh", "bm": "bam", "ee": "ewe", "ha": "hau",
    "ig": "ibo", "kg": "kng", "kj": "kua", "lg": "lug", "ln": "lin", "lu": "lub",
    "mg": "plt", "nr": "nbl", "ny": "nya", "om": "gaz", "rn": "run", "rw": "kin",
    "sg": "sag", "sn": "sna", "so": "som", "ss": "ssw", "st": "sot", "sw": "swh",
    "ti": "tir", "tn": "tsn", "ts": "tso", "tw": "twi", "ve": "ven", "wo": "wol",
    "xh": "xho", "yo": "yor", "zu": "zul",
}
DOMAIN_PRETTY = {"bible": "Bible", "watchtower": "Watchtower", "web": "Web"}


def words(text: str) -> int:
    return len(text.split())


def stats(docs: list[str]) -> dict[str, int]:
    return {
        "n_docs": len(docs),
        "word_count": sum(words(d) for d in docs),
        "char_count": sum(len(d) for d in docs),
        "byte_count": sum(len(d.encode("utf-8")) for d in docs),
    }


def write_jsonl_gz(path: Path, docs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")


def sample_to_words(docs: list[str], budget: int, seed: int) -> list[str]:
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    out, total = [], 0
    for i in order:
        if total >= budget:
            break
        out.append(docs[i])
        total += words(docs[i])
    return out


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def use_repo(repo: str) -> dict[str, "ac.Language"]:
    ac.HF_REPO_ID = repo
    ac.refresh()
    african, _ = ac.list_languages()
    return {CODE_MAP.get(lang.code, lang.code): lang for lang in african}


def _file_label(rel: str) -> str:
    """'Hausa_Arabic_script_hau_v3307.csv' -> 'Hausa Arabic script'."""
    return re.sub(r"_[^_]+_v\d+\.csv$", "", rel.split("/")[-1]).replace("_", " ")


def main_script_files(lang: "ac.Language") -> list[str]:
    """Files of the language's main written form: drop versions labelled as an alternative
    script ("Hausa Arabic script") when an unlabelled one exists; if every version is
    script-labelled ("Susu (Arabic Script)" / "Susu (Roman script)"), prefer Roman/Latin."""
    files = lang.files
    plain = [f for f in files if "script" not in _file_label(f).lower()]
    if plain:
        return plain
    latin = [f for f in files if re.search(r"roman|latin", _file_label(f), re.I)]
    return latin or files


def base_name(lang: "ac.Language") -> str:
    return min((_file_label(f) for f in main_script_files(lang)), key=len)


def bible_versions(lang: "ac.Language") -> list[tuple[str, list[dict[str, str]]]]:
    """(file, rows) per translation of the language's main written form, largest first."""
    files = main_script_files(lang)
    out = []
    for rel in files:
        path = hf_hub_download(BIBLE_REPO, rel, repo_type="dataset")
        rows = [r for r in read_csv_rows(path) if (r.get("local") or "").strip()]
        out.append((rel, rows))
    return sorted(out, key=lambda x: -len(x[1]))


def chapter_docs(rows: list[dict[str, str]], text_col: str) -> list[str]:
    chapters: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        key = r["verse_key"]
        chapters[key.rsplit(".", 1)[0]].append(r[text_col].strip())
    return [" ".join(v) for v in chapters.values()]


def jw_docs(lang: "ac.Language") -> list[str]:
    articles: dict[str, list[str]] = defaultdict(list)
    for rel in lang.files:
        path = hf_hub_download(JW_REPO, rel, repo_type="dataset")
        for r in read_csv_rows(path):
            text = (r.get("local") or "").strip()
            if text:
                articles[r["verse_key"].split("#", 1)[0]].append(text)
    return ["\n".join(v) for v in articles.values()]


def web_docs(lang: "ac.Language", budget: int) -> list[str]:
    """Stream the clean (@1) MADLAD file until ~3x the word budget is read."""
    clean = [f for f in lang.files if f.endswith("_v1.csv")] or lang.files
    fs = HfFileSystem()
    docs, total = [], 0
    with fs.open(f"datasets/{WEB_REPO}/{clean[0]}", "r", encoding="utf-8") as f:
        for r in csv.DictReader(line.replace("\0", "") for line in f):
            text = (r.get("local") or "").replace("\\n", "\n").strip()
            if text:
                docs.append(text)
                total += words(text)
                if total >= 3 * budget:
                    break
    return docs


def pristine_twi_docs(budget: int) -> list[str]:
    import pyarrow.parquet as pq

    repo, rev = PRISTINE_TWI
    path = hf_hub_download(repo, "data/train-00000-of-00004.parquet", repo_type="dataset",
                           revision=rev)
    docs, total = [], 0
    for text in pq.read_table(path, columns=["twi"]).column("twi").to_pylist():
        if text and text.strip():
            docs.append(text.strip())
            total += words(text)
            if total >= 3 * budget:
                break
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--langs", nargs="*", default=None, help="africa-corpus codes (default: all)")
    ap.add_argument("--min-sentences", type=int, default=10_000)
    ap.add_argument("--words-per-domain", type=int, default=200_000)
    ap.add_argument("--parity-verses", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out: Path = args.out
    budget = args.words_per_domain

    jw = use_repo(JW_REPO)
    web = use_repo(WEB_REPO)
    bible = use_repo(BIBLE_REPO)
    english_rows = read_csv_rows(hf_hub_download(BIBLE_REPO, "english_cache.csv",
                                                 repo_type="dataset"))
    english = {r["verse_key"]: r["eng"].strip() for r in english_rows if r.get("eng")}

    codes = args.langs or sorted(bible)
    meta: dict[str, dict] = {}
    table = []
    for n, code in enumerate(codes, 1):
        lang = bible[code]
        versions = bible_versions(lang)
        sentences = len({r["local"].strip() for _, rows in versions for r in rows})
        if sentences < args.min_sentences:
            print(f"[{n}/{len(codes)}] {code}: {sentences} sentences < {args.min_sentences}, skip")
            continue
        main_file, rows = versions[0]
        domains: dict[str, tuple[list[str], list[str]]] = {
            "bible": (sample_to_words(chapter_docs(rows, "local"), budget, args.seed),
                      [BIBLE_REPO])
        }
        extra = []
        if code in jw:
            extra.append(("watchtower", lambda: jw_docs(jw[code]), JW_REPO))
        if code == "twi":
            extra.append(("web", lambda: pristine_twi_docs(budget), PRISTINE_TWI[0]))
        elif code in web:
            extra.append(("web", lambda: web_docs(web[code], budget), WEB_REPO))
        for dom, load, source in extra:
            try:
                domains[dom] = (sample_to_words(load(), budget, args.seed), [source])
            except Exception as exc:  # one bad source file must not stop the whole build
                print(f"  {code}: skipping {dom} ({type(exc).__name__}: {exc})", flush=True)

        entry = {"name": code, "pretty_name": base_name(lang), "domains": [],
                 "parities": []}
        for dom, (docs, sources) in domains.items():
            if not docs:
                continue
            rel = f"{code}/{dom}.jsonl.gz"
            write_jsonl_gz(out / rel, docs)
            entry["domains"].append({"filepath": rel, "name": dom,
                                     "pretty_name": DOMAIN_PRETTY[dom], **stats(docs),
                                     "hf_datasets": sources})

        verses = {r["verse_key"]: r["local"].strip() for r in rows}
        shared = sorted(set(verses) & set(english), key=ac._verse_sort_key)
        random.Random(args.seed).shuffle(shared)
        shared = sorted(shared[: args.parity_verses], key=ac._verse_sort_key)
        if shared:
            tgt = [verses[k] for k in shared]
            ref = [english[k] for k in shared]
            write_jsonl_gz(out / f"{code}/parity.jsonl.gz", tgt)
            write_jsonl_gz(out / f"en/parity_{code}.jsonl.gz", ref)
            t, r = stats(tgt), stats(ref)
            entry["parities"].append({
                "filepath": f"{code}/parity.jsonl.gz", "name": "parity_en",
                "pretty_name": "Parity (English)", **t,
                "reference_filepath": f"en/parity_{code}.jsonl.gz", "reference_lang": "en",
                "reference_lang_pretty_name": "English",
                "reference_word_count": r["word_count"], "reference_char_count": r["char_count"],
                "reference_byte_count": r["byte_count"], "hf_datasets": [BIBLE_REPO],
            })
        totals = defaultdict(int)
        for d in entry["domains"]:
            for k in ("n_docs", "word_count", "char_count", "byte_count"):
                totals[k] += d[k]
        entry.update(totals)
        meta[code] = entry
        table.append({"code": code, "name": entry["pretty_name"], "sentences": sentences,
                      "bible_file": main_file, "domains": "+".join(domains)})
        print(f"[{n}/{len(codes)}] {code} ({entry['pretty_name']}): {sentences} sentences, "
              f"domains {list(domains)}, parity {len(shared)}", flush=True)

    # English reference language: Bible + Watchtower English.
    en_docs = {"bible": (sample_to_words(chapter_docs(english_rows, "eng"), budget, args.seed),
                         [BIBLE_REPO])}
    jw_en = read_csv_rows(hf_hub_download(JW_REPO, "english_cache.csv", repo_type="dataset"))
    arts: dict[str, list[str]] = defaultdict(list)
    for r in jw_en:
        if (r.get("eng") or "").strip():
            arts[r["verse_key"].split("#", 1)[0]].append(r["eng"].strip())
    en_docs["watchtower"] = (sample_to_words(["\n".join(v) for v in arts.values()], budget,
                                             args.seed), [JW_REPO])
    en = {"name": "en", "pretty_name": "English", "domains": [], "parities": None}
    for dom, (docs, sources) in en_docs.items():
        write_jsonl_gz(out / f"en/{dom}.jsonl.gz", docs)
        en["domains"].append({"filepath": f"en/{dom}.jsonl.gz", "name": dom,
                              "pretty_name": DOMAIN_PRETTY[dom], **stats(docs),
                              "hf_datasets": sources})
    for k in ("n_docs", "word_count", "char_count", "byte_count"):
        en[k] = sum(d[k] for d in en["domains"])
    meta["en"] = en

    (out / "dataset_meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True,
                                                          sort_keys=True))
    with open(out / "languages.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "name", "sentences", "bible_file", "domains"])
        w.writeheader()
        w.writerows(table)
    print(f"done: {len(meta) - 1} African languages + English -> {out}")


if __name__ == "__main__":
    main()
