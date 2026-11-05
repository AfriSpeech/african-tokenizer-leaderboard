"""Pair every African language with the high-resource language it is mainly used alongside.

African languages are almost always written and spoken next to English, French, Arabic,
Portuguese or Spanish, so the leaderboard ranks a language together with its companion: a
tokenizer is only useful for Twi if it also handles the English around it.

Method: the countries each language is used in come from the Africa Corpus Builder's
YouVersion metadata (youversion_africa_versions.csv). The metadata lists countries without
saying where a language is mainly spoken, so each country is weighted by its population
(COUNTRY_POP_M, approximate, millions): Yoruba listed for Nigeria and Benin goes with Nigeria.
Each country maps to its dominant high-resource language (COUNTRY_COMPANION) and the language
takes the companion with the most weight. OVERRIDES fixes cases the country
rule gets wrong. Output: benchmark/companion_languages.csv (code, name, countries, companion,
basis), reviewed by hand; edit OVERRIDES and re-run to change a pairing.

Usage: python benchmark/build_companions.py VERSIONS_CSV DATASET_META_YAML
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# Dominant high-resource language per country (official status and everyday use in
# education, government and written media).
COUNTRY_COMPANION = {
    # English
    "GH": "en", "NG": "en", "KE": "en", "UG": "en", "TZ": "en", "ZA": "en", "ZM": "en",
    "ZW": "en", "MW": "en", "BW": "en", "NA": "en", "LS": "en", "SZ": "en", "SL": "en",
    "LR": "en", "GM": "en", "SS": "en", "RW": "en", "ET": "en", "ER": "en", "SH": "en",
    # French
    "SN": "fr", "ML": "fr", "BF": "fr", "NE": "fr", "GN": "fr", "CI": "fr", "TG": "fr",
    "BJ": "fr", "CM": "fr", "CF": "fr", "CG": "fr", "CD": "fr", "GA": "fr", "TD": "fr",
    "BI": "fr", "DJ": "fr", "KM": "fr", "MG": "fr", "MU": "fr", "SC": "fr", "RE": "fr",
    "YT": "fr",
    # Arabic
    "EG": "ar", "LY": "ar", "TN": "ar", "DZ": "ar", "MA": "ar", "EH": "ar", "MR": "ar",
    "SD": "ar", "SO": "ar",
    # Portuguese
    "AO": "pt", "MZ": "pt", "GW": "pt", "CV": "pt", "ST": "pt",
    # Spanish
    "GQ": "es",
}

# Approximate population, millions (2024): a proxy for where a multi-country language is
# mainly used.
COUNTRY_POP_M = {
    "NG": 224, "ET": 127, "EG": 113, "CD": 102, "TZ": 67, "ZA": 61, "KE": 55, "UG": 48,
    "SD": 48, "DZ": 46, "MA": 37, "AO": 37, "GH": 34, "MZ": 34, "MG": 30, "CI": 29, "CM": 28,
    "NE": 27, "ML": 23, "BF": 23, "MW": 21, "ZM": 20, "TD": 18, "SO": 18, "SN": 18, "ZW": 16,
    "GN": 14, "RW": 14, "BJ": 13, "BI": 13, "TN": 12, "SS": 11, "TG": 9, "SL": 9, "LY": 7,
    "CG": 6, "LR": 5, "CF": 5, "MR": 5, "ER": 4, "NA": 3, "GM": 3, "BW": 3, "GA": 2.4,
    "LS": 2.3, "GW": 2.1, "GQ": 1.7, "MU": 1.3, "SZ": 1.2, "DJ": 1.1, "RE": 0.9, "KM": 0.8,
    "CV": 0.6, "EH": 0.6, "ST": 0.2, "YT": 0.3, "SC": 0.1, "SH": 0.01,
}

# Languages the country rule gets wrong (code -> (companion, reason)).
OVERRIDES = {
    "swh": ("en", "East African lingua franca; English in all its main countries"),
    "hau": ("en", "Mostly Nigeria (English); Niger is the minority"),
    "som": ("en", "English dominates Somali written and digital use (Arabic is co-official)"),
    "ary": ("ar", "Moroccan Arabic: Arabic"),
    "arq": ("ar", "Algerian Arabic: Arabic"),
    "apd": ("ar", "Sudanese Arabic: Arabic"),
    "shu": ("ar", "Chadian Arabic: Arabic"),
    "afr": ("en", "South Africa and Namibia"),
    "twi": ("en", "Ghana (no country in the metadata)"),
    "fat": ("en", "Ghana (no country in the metadata)"),
    "aka": ("en", "Ghana"),
    "kri": ("en", "Sierra Leone"),
    "run": ("fr", "Burundi"),
    "knk": ("en", "Mainly Sierra Leone"),
    "lon": ("en", "Malawi Lomwe: Malawi"),
    "swk": ("en", "Malawi Sena: Malawi"),
    "nyu": ("pt", "Nyungwe: Tete, Mozambique"),
    "mhw": ("en", "Mainly Namibia"),
}


def main(versions_csv: str, meta_yaml: str) -> None:
    weights: dict[str, Counter] = defaultdict(Counter)
    names: dict[str, str] = {}
    for r in csv.DictReader(open(versions_csv, encoding="utf-8")):
        names.setdefault(r["lang_code"], r["lang_name"])
        for c in (r["country"] or "").split(";"):
            if c.strip():
                weights[r["lang_code"]][c.strip()] += 1

    meta = yaml.safe_load(open(meta_yaml, encoding="utf-8"))
    african = sorted(c for c, m in meta.items() if m.get("group", "african") == "african")
    rows, review = [], []
    for code in african:
        countries = weights.get(code, Counter())
        if code in OVERRIDES:
            comp, basis = OVERRIDES[code][0], f"override: {OVERRIDES[code][1]}"
        else:
            votes: Counter = Counter()
            for country, w in countries.items():
                if country in COUNTRY_COMPANION:
                    votes[COUNTRY_COMPANION[country]] += COUNTRY_POP_M.get(country, 1)
            if not votes:
                comp, basis = "en", "no African country on record; default English"
                review.append((code, "no country"))
            else:
                (comp, top), *rest = votes.most_common()
                basis = "countries (population-weighted): " + ", ".join(
                    f"{c}→{COUNTRY_COMPANION.get(c, '?')}" for c in sorted(
                        countries, key=lambda c: -COUNTRY_POP_M.get(c, 0)))
                if rest and rest[0][1] > 0.6 * top:
                    review.append((code, f"close: {dict(votes)}"))
                    basis += " (close; review)"
        rows.append({"code": code, "name": meta[code]["pretty_name"],
                     "countries": ";".join(c for c, _ in countries.most_common()),
                     "companion": comp, "basis": basis})

    out = Path(__file__).parent / "companion_languages.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "name", "countries", "companion", "basis"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} languages, companions {dict(Counter(r['companion'] for r in rows))}")
    for code, why in review:
        print(f"  review {code} ({meta[code]['pretty_name']}): {why}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
