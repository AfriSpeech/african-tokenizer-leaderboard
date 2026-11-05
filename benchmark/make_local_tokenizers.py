"""Wrap raw tokenizer.json files (Pocket TTS) as AutoTokenizer-loadable folders.

  tokenizers/pocket-tts-english   Kyutai's released English Pocket TTS tokenizer
  tokenizers/pocket-tts-en-twi    English + Twi merged tokenizer (GhanaOpenAI/pocket-tts-twi)
  tokenizers/serengeti            UBC-NLP/serengeti's own tokenizer.json; its tokenizer_config
                                  stores mask_token in an old dict format that current
                                  transformers rejects, so only the config is rewritten

Usage: python benchmark/make_local_tokenizers.py [--en-twi PATH_TO_en_twi_4000.json]
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

KYUTAI = ("kyutai/pocket-tts-without-voice-cloning",
          "languages/english_2026-04_24l/tokenizer.json",
          "00eac05ed3d16bdc3f6b5d598874019c34a89214")
EN_TWI_URL = ("https://github.com/GhanaOpenAI/pocket-tts-twi/blob/twi-merged-tokenizer/"
              "training/recipes/twi/tokenizer/en_twi_4000.json")


def wrap(src: Path, dst: Path, note: str, **extra: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst / "tokenizer.json")
    (dst / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>",
        "add_bos_token": False, "add_eos_token": False, "model_max_length": 1_000_000,
        **extra,
    }, indent=1))
    (dst / "README.md").write_text(note + "\n")
    print("wrote", dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-twi", type=Path, required=True,
                    help=f"en_twi_4000.json (from {EN_TWI_URL})")
    ap.add_argument("--out", type=Path, default=Path("tokenizers"))
    args = ap.parse_args()
    repo, path, rev = KYUTAI
    wrap(Path(hf_hub_download(repo, path, revision=rev)), args.out / "pocket-tts-english",
         f"Kyutai Pocket TTS English tokenizer ({repo}/{path}@{rev}).")
    wrap(args.en_twi, args.out / "pocket-tts-en-twi",
         f"Pocket TTS English+Twi merged tokenizer ({EN_TWI_URL}).")
    wrap(Path(hf_hub_download("UBC-NLP/serengeti", "tokenizer.json")), args.out / "serengeti",
         "UBC-NLP/serengeti tokenizer.json, unchanged; config rewritten for current "
         "transformers (the original stores mask_token as a dict).",
         mask_token="<mask>", cls_token="<s>", sep_token="</s>")


if __name__ == "__main__":
    main()
