---
task_categories:
  - text-generation
language:
  - en
  - eng
  - pt
  - por
  - es
  - spa
  - fr
  - fra
  - zh
  - cmn
  - ja
  - jpn
  - de
  - deu
  - ru
  - rus
  - ar
  - arb
  - it
  - ita
  - pl
  - pol
  - ko
  - kor
  - vi
  - vie
  - code
multilinguality:
  - multilingual
pretty_name: Multilingual Tokenizer Benchmark
size_categories:
- 100K<n<1M
configs:
  - config_name: default
    data_files:
    - split: train
      path: "**/*.jsonl.gz"
  - config_name: en
    data_files:
    - split: train
      path: en/*.jsonl.gz
  - config_name: pt
    data_files:
    - split: train
      path: pt/*.jsonl.gz
  - config_name: es
    data_files:
    - split: train
      path: es/*.jsonl.gz
  - config_name: fr
    data_files:
    - split: train
      path: fr/*.jsonl.gz
  - config_name: zh
    data_files:
    - split: train
      path: zh/*.jsonl.gz
  - config_name: ja
    data_files:
    - split: train
      path: ja/*.jsonl.gz
  - config_name: de
    data_files:
    - split: train
      path: de/*.jsonl.gz
  - config_name: ru
    data_files:
    - split: train
      path: ru/*.jsonl.gz
  - config_name: ar
    data_files:
    - split: train
      path: ar/*.jsonl.gz
  - config_name: it
    data_files:
    - split: train
      path: it/*.jsonl.gz
  - config_name: pl
    data_files:
    - split: train
      path: pl/*.jsonl.gz
  - config_name: ko
    data_files:
    - split: train
      path: ko/*.jsonl.gz
  - config_name: code
    data_files:
    - split: train
      path: code/*.jsonl.gz
  - config_name: vi
    data_files:
    - split: train
      path: vi/*.jsonl.gz
source_datasets:
  - bigcode/the-stack-smol
  - badranx/opus_raw
  - HuggingFaceFW/fineweb-2
  - sebastiandizon/genius-song-lyrics
  - rntc/pubmed_preprocess
  - wikimedia/wikipedia
  - joelniklaus/Multi_Legal_Pile
  - almanach/HALvest
  - premio-ai/TheArabicPile_Medical
  - ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts
  - asas-ai/financial_news
  - bigscience-data/roots_ar_ted_talks_iwslt
  - premio-ai/TheArabicPile_Lyrics
  - AIR-Bench/qa_news_ar
  - premio-ai/TheArabicPile_Poetry
  - bigscience-data/roots_ar_wikipedia
  - bigcode/starcoderdata
  - AIR-Bench/qa_news_de
  - orionweller/books_mds_incremental
  - ElKulako/stocktwits-emoji
  - Orion-zhen/dpo-codealpaca-emoji
  - Orion-zhen/dpo-mathinstuct-emoji
  - Orion-zhen/dpo-physical-reasoning-emoji
  - AIR-Bench/qa_finance_en
  - bigscience-data/roots_en_ted_talks_iwslt
  - orionweller/cc_news_mds_incremental
  - DanFosing/public-domain-poetry
  - orionweller/arxiv_mds_incremental
  - orionweller/pes2o_mds_incremental
  - orionweller/reddit_mds_incremental
  - HuggingFaceFW/fineweb
  - orionweller/wikipedia_mds_incremental
  - jorgeortizfuentes/spanish_books
  - bigscience-data/roots_es_ted_talks_iwslt
  - LeoCordoba/CC-NEWS-ES
  - andreamorgar/spanish_poetry
  - jorge-henao/disco_poetry_spanish
  - eduagarcia/scielo_abstracts
  - Arconte/Dominican_reddit_raw_corpus
  - bigscience-data/roots_es_wikipedia
  - AIR-Bench/qa_finance_fr
  - bigscience-data/roots_fr_ted_talks_iwslt
  - AIR-Bench/qa_news_fr
  - kamarko/ccnews-french-subset
  - manu/french_poetry
  - bigscience-data/roots_fr_wikipedia
  - HiTZ/Multilingual-Medical-Corpus
  - globis-university/aozorabunko-clean
  - geniacllm/hanrei
  - kajuma/CC-news-2024-July-October-cleaned
  - p1atdev/modern_haiku
  - kunishou/J-ResearchCorpus
  - p1atdev/open2ch
  - BCCard/BCCard-Finance-Kor-QnA
  - jihye-moon/klac_legal_aid_counseling
  - joonhok-exo-ai/korean_law_open_data_precedents
  - AIR-Bench/qa_news_ko
  - daekeun-ml/naver-news-summarization-ko
  - werty1248/Korea-Related-Reddit-comments
  - werty1248/Korea-Related-Reddit-posts
  - WiktorS/polish-news
  - winterkitsune/elka-pl-news
  - eduagarcia/pd_books_pt
  - eduagarcia/LegalPT_dedup
  - bigscience-data/roots_pt_ted_talks_iwslt
  - eduagarcia/cc_news_pt_v2
  - eduagarcia/scielo_papers
  - carolina-c4ai/corpus-carolina
  - tallesl/quinquilharia
  - bigscience-data/roots_pt_wikipedia
  - blinoff/medical_qa_ru_data
  - sevenreasons/genius-lyrics-russian
  - AIR-Bench/qa_news_ru
  - IlyaGusev/ru_news
  - 0x7o/poemma-10k
  - AnyaSchen/russian_poetry_with_keywords
  - mlsa-iai-msu-lab/ru_sci_bench
  - VietAI/vi_pubmed
  - bigscience-data/roots_vi_ted_talks_iwslt
  - bigscience-data/roots_vi_binhvq_news_corpus
  - bigscience-data/roots_vi_vietnamese_poetry
  - DavidLanz/medical_pretrain
  - Orion-zhen/dpo-emoji-zh
  - Orion-zhen/dpo-ruozhiba-emoji
  - shareAI/DPO-zh-en-emoji
  - Duxiaoman-DI/FinCorpus
  - TigerResearch/tigerbot-law-plugin
  - bigscience-data/roots_zh_ted_talks_iwslt
  - dirtycomputer/chinese_lyrics
  - AIR-Bench/qa_news_zh
  - liswei/news-collection-zhtw
  - erhwenkuo/poetry-chinese-zhtw
  - bigscience-data/roots_zh-cn_wikipedia
---
# Multilingual Tokenizer Benchmark


More details of each subset like word count, character count, original sources, etc, can be found in the [dataset_meta.yaml](dataset_meta.yaml) file in the repository root.


### Natural language word count functions
Download spacy models
```bash
pip install ntlk spacy pygments underthesea camel-tools

python -m spacy download ko_core_news_sm
python -m spacy download ja_core_news_sm
python -m spacy download zh_core_web_sm
```

```python
import nltk
nltk.download('punkt_tab')
nltk.download('words')
from nltk import word_tokenize as word_tokenize_nltk
from underthesea import word_tokenize as word_tokenize_underthesea
from camel_tools.tokenizers.word import simple_word_tokenize as word_tokenize_camel_tools
import spacy

langs_nltk = {
    'en': 'english',
    'pt': 'portuguese',
    'it': 'italian',
    'pl': 'polish',
    'de': 'german',
    'es': 'spanish',
    'fr': 'french',
    'ru': 'russian',
}

langs_spacy = {
    'ja': spacy.load('ja_core_news_sm', disable=['parser', 'ner', 'lemmatizer', 'tagger', 'attribute_ruler']),
    'zh': spacy.load('zh_core_web_sm', disable=['parser', 'ner', 'lemmatizer', 'tagger', 'attribute_ruler']),
    'ko': spacy.load('ko_core_news_sm', disable=['parser', 'ner', 'lemmatizer', 'tagger', 'attribute_ruler']),
}

def word_tokenize_spacy(text, model):
    doc = model(text)
    return [token.text for token in doc if not token.is_space]

def word_tokenize(text, language):
    if language in langs_spacy:
        return word_tokenize_spacy(text, langs_spacy[language])
    elif language in langs_nltk:
        return word_tokenize_nltk(text, langs_nltk[language])
    elif language == 'vi':
        return word_tokenize_underthesea(text)
    elif language == 'ar':
        return word_tokenize_camel_tools(text)
    else:
        raise ValueError(f"Language {language} not supported")

#### Usage Example ####
word_count_pt = len(word_tokenize("Olá Mundo.", 'pt'))
word_count_ko = len(word_tokenize("이것은 한국어 문장입니다.", 'ko'))
```

### Code word count function
```python
import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize

from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

NATURAL_LANGUAGE_PARENT_TOKEN_TYPES = {
    Token.Comment,
    Token.Literal.String,
    Token.Text,
    Token.Generic.Heading,    # For Markdown,
    Token.Generic.Subheading, # For Markdown,
    Token.Generic.Emph,       # For Markdown *italic*
    Token.Generic.Strong,     # For Markdown **bold**
    Token.Generic.Output,     # For shell output,
    Token.Generic.Error,      # Error messages
    Token.Generic.Traceback,  # Traceback messages
}

def _is_natural_language_like(tok_type) -> bool:
    for nl_parent_type in NATURAL_LANGUAGE_PARENT_TOKEN_TYPES:
        if tok_type in nl_parent_type: # This checks if tok_type is nl_parent_type OR a child
            return True
    return False

def pygments_tokenize(code: str, language: str) -> list[str]:
    lexer = get_lexer_by_name(language, stripnl=True, stripall=True)
    tokens = lex(code, lexer)

    result = []

    for tok_type, value in tokens:
        if _is_natural_language_like(tok_type):
            # Recursively tokenize comment or string content. Use the default english tokenizer.
            result.extend(word_tokenize(value))
        else:
            # Tokenize symbol/keyword/identifier at surface level
            result.append(value)

    return [t for t in result if t.strip() != '']

#### Usage Example ####
word_count_json = len(pygments_tokenize('{"hello": 1}', 'json'))
word_count_python = len(pygments_tokenize("lambda x: print('hello')", "python"))
```

### Per language stats

| Code   | Language              |   Num. of Domains |   Num. of Docs |   Num. of Characters |   Num. of Words |   Num. of Bytes | Data Types                                                                                                                       |
|:-------|:----------------------|------------------:|---------------:|---------------------:|----------------:|----------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| code   | Programming Languages |                18 |           4260 |             17454895 |         3645230 |        17621590 | C, C++, CSS, Dockerfile, Go, HTML, JSON, Java, JavaScript, Lua, Markdown, PHP, Python, Rust, SQL, Shell, TeX, TypeScript                         |
| en     | English               |                14 |           8513 |             14087244 |         2812929 |        14177905 | Biomedical, Books, Emoji Heavy, Finance, Legal, Live Speech, Lyrics, News, Poetry, Scientific Papers, Social Networks, Subtitles, Web, Wikipedia |
| es     | Spanish               |                12 |           7739 |             12786745 |         2408966 |        13063395 | Biomedical, Books, Legal, Live Speech, Lyrics, News, Poetry, Scientific Papers, Social Networks, Subtitles, Web, Wikipedia                       |
| fr     | French                |                11 |           6812 |             11477784 |         2206430 |        11867606 | Biomedical, Finance, Legal, Live Speech, Lyrics, News, Poetry, Scientific Papers, Subtitles, Web, Wikipedia                                      |
| pt     | Portuguese            |                11 |           7709 |             11546654 |         2207307 |        11887461 | Biomedical, Books, Legal, Live Speech, Lyrics, News, Scientific Papers, Social Networks, Subtitles, Web, Wikipedia                               |
| zh     | Chinese               |                11 |          10769 |              3999493 |         2213799 |        10471916 | Biomedical, Emoji Heavy, Finance, Legal, Live Speech, Lyrics, News, Poetry, Subtitles, Web, Wikipedia                                            |
| ar     | Arabic                |                10 |          12410 |              9909750 |         2005226 |        17685166 | Biomedical, Books, Finance, Live Speech, Lyrics, News, Poetry, Subtitles, Web, Wikipedia                                                         |
| ja     | Japanese              |                10 |          11039 |              3801952 |         2008396 |         9717840 | Books, Legal, Lyrics, News, Poetry, Scientific Papers, Social Networks, Subtitles, Web, Wikipedia                                                |
| de     | German                |                 8 |           5119 |             11016640 |         1841622 |        11207161 | Biomedical, Legal, Lyrics, News, Scientific Papers, Subtitles, Web, Wikipedia                                                                    |
| ko     | Korean                |                 8 |           9498 |              6060455 |         1602826 |        12532955 | Finance, Legal, Lyrics, News, Social Networks, Subtitles, Web, Wikipedia                                                                         |
| ru     | Russian               |                 8 |           8729 |              9105314 |         1612537 |        16277665 | Biomedical, Lyrics, News, Poetry, Scientific Papers, Subtitles, Web, Wikipedia                                                                   |
| vi     | Vietnamese            |                 8 |           8973 |              8533015 |         1602500 |        10911187 | Biomedical, Live Speech, Lyrics, News, Poetry, Subtitles, Web, Wikipedia                                                                         |
| it     | Italian               |                 7 |           6976 |              7706475 |         1408363 |         7792425 | Biomedical, Legal, Lyrics, Scientific Papers, Subtitles, Web, Wikipedia                                                                          |
| pl     | Polish                |                 6 |           3990 |              7278038 |         1204690 |         7694868 | Legal, Lyrics, News, Subtitles, Web, Wikipedia                                                                                                   |

### Sourced datasets:

|    | HF Dataset                                                                                                                                  |   Num. Subsets | Languages                                      | Data Types                                                                                                          |
|---:|:---------------------------------------------------------------------------------------------------------------------------------------------------|--------:|:---------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
|  0 | [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol)                                                                   |      17 | code                                               | C, C++, CSS, Dockerfile, Go, HTML, Java, JavaScript, Lua, Markdown, PHP, Python, Rust, SQL, Shell, TeX, TypeScript |
|  1 | [badranx/opus_raw](https://huggingface.co/datasets/badranx/opus_raw)                                                                               |      14 | ar, de, en, es, fr, it, ja, ko, pl, pt, ru, vi, zh | Subtitles                                                                                                          |
|  2 | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)                                                                 |      12 | ar, de, es, fr, it, ja, ko, pl, pt, ru, vi, zh     | Web                                                                                                                |
|  3 | [sebastiandizon/genius-song-lyrics](https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics)                                             |      11 | de, en, es, fr, it, ja, ko, pl, pt, vi, zh         | Lyrics                                                                                                             |
|  4 | [rntc/pubmed_preprocess](https://huggingface.co/datasets/rntc/pubmed_preprocess)                                                                   |       8 | de, en, es, fr, it, pt, ru, zh                     | Biomedical                                                                                                         |
|  5 | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)                                                                         |       7 | de, it, ja, ko, pl, ru, vi                         | Wikipedia                                                                                                          |
|  6 | [joelniklaus/Multi_Legal_Pile](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile)                                                       |       6 | de, en, es, fr, it, pl                             | Legal                                                                                                              |
|  7 | [almanach/HALvest](https://huggingface.co/datasets/almanach/HALvest)                                                                               |       5 | de, es, fr, it, ru                                 | Scientific Papers                                                                                                  |
|  8 | [premio-ai/TheArabicPile_Medical](https://huggingface.co/datasets/premio-ai/TheArabicPile_Medical)                                                 |       1 | ar                                                 | Biomedical                                                                                                         |
|  9 | [ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts](https://huggingface.co/datasets/ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts) |       1 | ar                                                 | Books                                                                                                              |
| 10 | [asas-ai/financial_news](https://huggingface.co/datasets/asas-ai/financial_news)                                                                   |       1 | ar                                                 | Finance                                                                                                            |
| 11 | [bigscience-data/roots_ar_ted_talks_iwslt](https://huggingface.co/datasets/bigscience-data/roots_ar_ted_talks_iwslt)                               |       1 | ar                                                 | Live Speech                                                                                                        |
| 12 | [premio-ai/TheArabicPile_Lyrics](https://huggingface.co/datasets/premio-ai/TheArabicPile_Lyrics)                                                   |       1 | ar                                                 | Lyrics                                                                                                             |
| 13 | [AIR-Bench/qa_news_ar](https://huggingface.co/datasets/AIR-Bench/qa_news_ar)                                                                       |       1 | ar                                                 | News                                                                                                               |
| 14 | [premio-ai/TheArabicPile_Poetry](https://huggingface.co/datasets/premio-ai/TheArabicPile_Poetry)                                                   |       1 | ar                                                 | Poetry                                                                                                             |
| 15 | [bigscience-data/roots_ar_wikipedia](https://huggingface.co/datasets/bigscience-data/roots_ar_wikipedia)                                           |       1 | ar                                                 | Wikipedia                                                                                                          |
| 16 | [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)                                                                     |       1 | code                                               | JSON                                                                                                               |
| 17 | [AIR-Bench/qa_news_de](https://huggingface.co/datasets/AIR-Bench/qa_news_de)                                                                       |       1 | de                                                 | News                                                                                                               |
| 18 | [orionweller/books_mds_incremental](https://huggingface.co/datasets/orionweller/books_mds_incremental)                                             |       1 | en                                                 | Books                                                                                                              |
| 19 | [ElKulako/stocktwits-emoji](https://huggingface.co/datasets/ElKulako/stocktwits-emoji)                                                             |       1 | en                                                 | Emoji Heavy                                                                                                        |
| 20 | [Orion-zhen/dpo-codealpaca-emoji](https://huggingface.co/datasets/Orion-zhen/dpo-codealpaca-emoji)                                                 |       1 | en                                                 | Emoji Heavy                                                                                                        |
| 21 | [Orion-zhen/dpo-mathinstuct-emoji](https://huggingface.co/datasets/Orion-zhen/dpo-mathinstuct-emoji)                                               |       1 | en                                                 | Emoji Heavy                                                                                                        |
| 22 | [Orion-zhen/dpo-physical-reasoning-emoji](https://huggingface.co/datasets/Orion-zhen/dpo-physical-reasoning-emoji)                                 |       1 | en                                                 | Emoji Heavy                                                                                                        |
| 23 | [AIR-Bench/qa_finance_en](https://huggingface.co/datasets/AIR-Bench/qa_finance_en)                                                                 |       1 | en                                                 | Finance                                                                                                            |
| 24 | [bigscience-data/roots_en_ted_talks_iwslt](https://huggingface.co/datasets/bigscience-data/roots_en_ted_talks_iwslt)                               |       1 | en                                                 | Live Speech                                                                                                        |
| 25 | [orionweller/cc_news_mds_incremental](https://huggingface.co/datasets/orionweller/cc_news_mds_incremental)                                         |       1 | en                                                 | News                                                                                                               |
| 26 | [DanFosing/public-domain-poetry](https://huggingface.co/datasets/DanFosing/public-domain-poetry)                                                   |       1 | en                                                 | Poetry                                                                                                             |
| 27 | [orionweller/arxiv_mds_incremental](https://huggingface.co/datasets/orionweller/arxiv_mds_incremental)                                             |       1 | en                                                 | Scientific Papers                                                                                                  |
| 28 | [orionweller/pes2o_mds_incremental](https://huggingface.co/datasets/orionweller/pes2o_mds_incremental)                                             |       1 | en                                                 | Scientific Papers                                                                                                  |
| 29 | [orionweller/reddit_mds_incremental](https://huggingface.co/datasets/orionweller/reddit_mds_incremental)                                           |       1 | en                                                 | Social Networks                                                                                                    |
| 30 | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)                                                                     |       1 | en                                                 | Web                                                                                                                |
| 31 | [orionweller/wikipedia_mds_incremental](https://huggingface.co/datasets/orionweller/wikipedia_mds_incremental)                                     |       1 | en                                                 | Wikipedia                                                                                                          |
| 32 | [jorgeortizfuentes/spanish_books](https://huggingface.co/datasets/jorgeortizfuentes/spanish_books)                                                 |       1 | es                                                 | Books                                                                                                              |
| 33 | [bigscience-data/roots_es_ted_talks_iwslt](https://huggingface.co/datasets/bigscience-data/roots_es_ted_talks_iwslt)                               |       1 | es                                                 | Live Speech                                                                                                        |
| 34 | [LeoCordoba/CC-NEWS-ES](https://huggingface.co/datasets/LeoCordoba/CC-NEWS-ES)                                                                     |       1 | es                                                 | News                                                                                                               |
| 35 | [andreamorgar/spanish_poetry](https://huggingface.co/datasets/andreamorgar/spanish_poetry)                                                         |       1 | es                                                 | Poetry                                                                                                             |
| 36 | [jorge-henao/disco_poetry_spanish](https://huggingface.co/datasets/jorge-henao/disco_poetry_spanish)                                               |       1 | es                                                 | Poetry                                                                                                             |
| 37 | [eduagarcia/scielo_abstracts](https://huggingface.co/datasets/eduagarcia/scielo_abstracts)                                                         |       1 | es                                                 | Scientific Papers                                                                                                  |
| 38 | [Arconte/Dominican_reddit_raw_corpus](https://huggingface.co/datasets/Arconte/Dominican_reddit_raw_corpus)                                         |       1 | es                                                 | Social Networks                                                                                                    |
| 39 | [bigscience-data/roots_es_wikipedia](https://huggingface.co/datasets/bigscience-data/roots_es_wikipedia)                                           |       1 | es                                                 | Wikipedia                                                                                                          |
| 40 | [AIR-Bench/qa_finance_fr](https://huggingface.co/datasets/AIR-Bench/qa_finance_fr)                                                                 |       1 | fr                                                 | Finance                                                                                                            |
| 41 | [bigscience-data/roots_fr_ted_talks_iwslt](https://huggingface.co/datasets/bigscience-data/roots_fr_ted_talks_iwslt)                               |       1 | fr                                                 | Live Speech                                                                                                        |
| 42 | [AIR-Bench/qa_news_fr](https://huggingface.co/datasets/AIR-Bench/qa_news_fr)                                                                       |       1 | fr                                                 | News                                                                                                               |
| 43 | [kamarko/ccnews-french-subset](https://huggingface.co/datasets/kamarko/ccnews-french-subset)                                                       |       1 | fr                                                 | News                                                                                                               |
| 44 | [manu/french_poetry](https://huggingface.co/datasets/manu/french_poetry)                                                                           |       1 | fr                                                 | Poetry                                                                                                             |
| 45 | [bigscience-data/roots_fr_wikipedia](https://huggingface.co/datasets/bigscience-data/roots_fr_wikipedia)                                           |       1 | fr                                                 | Wikipedia                                                                                                          |
| 46 | [HiTZ/Multilingual-Medical-Corpus](https://huggingface.co/datasets/HiTZ/Multilingual-Medical-Corpus)                                               |       1 | it                                                 | Biomedical                                                                                                         |
| 47 | [globis-university/aozorabunko-clean](https://huggingface.co/datasets/globis-university/aozorabunko-clean)                                         |       1 | ja                                                 | Books                                                                                                              |
| 48 | [geniacllm/hanrei](https://huggingface.co/datasets/geniacllm/hanrei)                                                                               |       1 | ja                                                 | Legal                                                                                                              |
| 49 | [kajuma/CC-news-2024-July-October-cleaned](https://huggingface.co/datasets/kajuma/CC-news-2024-July-October-cleaned)                               |       1 | ja                                                 | News                                                                                                               |
| 50 | [p1atdev/modern_haiku](https://huggingface.co/datasets/p1atdev/modern_haiku)                                                                       |       1 | ja                                                 | Poetry                                                                                                             |
| 51 | [kunishou/J-ResearchCorpus](https://huggingface.co/datasets/kunishou/J-ResearchCorpus)                                                             |       1 | ja                                                 | Scientific Papers                                                                                                  |
| 52 | [p1atdev/open2ch](https://huggingface.co/datasets/p1atdev/open2ch)                                                                                 |       1 | ja                                                 | Social Networks                                                                                                    |
| 53 | [BCCard/BCCard-Finance-Kor-QnA](https://huggingface.co/datasets/BCCard/BCCard-Finance-Kor-QnA)                                                     |       1 | ko                                                 | Finance                                                                                                            |
| 54 | [jihye-moon/klac_legal_aid_counseling](https://huggingface.co/datasets/jihye-moon/klac_legal_aid_counseling)                                       |       1 | ko                                                 | Legal                                                                                                              |
| 55 | [joonhok-exo-ai/korean_law_open_data_precedents](https://huggingface.co/datasets/joonhok-exo-ai/korean_law_open_data_precedents)                   |       1 | ko                                                 | Legal                                                                                                              |
| 56 | [AIR-Bench/qa_news_ko](https://huggingface.co/datasets/AIR-Bench/qa_news_ko)                                                                       |       1 | ko                                                 | News                                                                                                               |
| 57 | [daekeun-ml/naver-news-summarization-ko](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko)                                   |       1 | ko                                                 | News                                                                                                               |
| 58 | [werty1248/Korea-Related-Reddit-comments](https://huggingface.co/datasets/werty1248/Korea-Related-Reddit-comments)                                 |       1 | ko                                                 | Social Networks                                                                                                    |
| 59 | [werty1248/Korea-Related-Reddit-posts](https://huggingface.co/datasets/werty1248/Korea-Related-Reddit-posts)                                       |       1 | ko                                                 | Social Networks                                                                                                    |
| 60 | [WiktorS/polish-news](https://huggingface.co/datasets/WiktorS/polish-news)                                                                         |       1 | pl                                                 | News                                                                                                               |
| 61 | [winterkitsune/elka-pl-news](https://huggingface.co/datasets/winterkitsune/elka-pl-news)                                                           |       1 | pl                                                 | News                                                                                                               |
| 62 | [eduagarcia/pd_books_pt](https://huggingface.co/datasets/eduagarcia/pd_books_pt)                                                                   |       1 | pt                                                 | Books                                                                                                              |
| 63 | [eduagarcia/LegalPT_dedup](https://huggingface.co/datasets/eduagarcia/LegalPT_dedup)                                                               |       1 | pt                                                 | Legal                                                                                                              |
| 64 | [bigscience-data/roots_pt_ted_talks_iwslt](https://huggingface.co/datasets/bigscience-data/roots_pt_ted_talks_iwslt)                               |       1 | pt                                                 | Live Speech                                                                                                        |
| 65 | [eduagarcia/cc_news_pt_v2](https://huggingface.co/datasets/eduagarcia/cc_news_pt_v2)                                                               |       1 | pt                                                 | News                                                                                                               |
| 66 | [eduagarcia/scielo_papers](https://huggingface.co/datasets/eduagarcia/scielo_papers)                                                               |       1 | pt                                                 | Scientific Papers                                                                                                  |
| 67 | [carolina-c4ai/corpus-carolina](https://huggingface.co/datasets/carolina-c4ai/corpus-carolina)                                                     |       1 | pt                                                 | Social Networks                                                                                                    |
| 68 | [tallesl/quinquilharia](https://huggingface.co/datasets/tallesl/quinquilharia)                                                                     |       1 | pt                                                 | Social Networks                                                                                                    |
| 69 | [bigscience-data/roots_pt_wikipedia](https://huggingface.co/datasets/bigscience-data/roots_pt_wikipedia)                                           |       1 | pt                                                 | Wikipedia                                                                                                          |
| 70 | [blinoff/medical_qa_ru_data](https://huggingface.co/datasets/blinoff/medical_qa_ru_data)                                                           |       1 | ru                                                 | Biomedical                                                                                                         |
| 71 | [sevenreasons/genius-lyrics-russian](https://huggingface.co/datasets/sevenreasons/genius-lyrics-russian)                                           |       1 | ru                                                 | Lyrics                                                                                                             |
| 72 | [AIR-Bench/qa_news_ru](https://huggingface.co/datasets/AIR-Bench/qa_news_ru)                                                                       |       1 | ru                                                 | News                                                                                                               |
| 73 | [IlyaGusev/ru_news](https://huggingface.co/datasets/IlyaGusev/ru_news)                                                                             |       1 | ru                                                 | News                                                                                                               |
| 74 | [0x7o/poemma-10k](https://huggingface.co/datasets/0x7o/poemma-10k)                                                                                 |       1 | ru                                                 | Poetry                                                                                                             |
| 75 | [AnyaSchen/russian_poetry_with_keywords](https://huggingface.co/datasets/AnyaSchen/russian_poetry_with_keywords)                                   |       1 | ru                                                 | Poetry                                                                                                             |
| 76 | [mlsa-iai-msu-lab/ru_sci_bench](https://huggingface.co/datasets/mlsa-iai-msu-lab/ru_sci_bench)                                                     |       1 | ru                                                 | Scientific Papers                                                                                                  |
| 77 | [VietAI/vi_pubmed](https://huggingface.co/datasets/VietAI/vi_pubmed)                                                                               |       1 | vi                                                 | Biomedical                                                                                                         |
| 78 | [bigscience-data/roots_vi_ted_talks_iwslt](https://huggingface.co/datasets/bigscience-data/roots_vi_ted_talks_iwslt)                               |       1 | vi                                                 | Live Speech                                                                                                        |
| 79 | [bigscience-data/roots_vi_binhvq_news_corpus](https://huggingface.co/datasets/bigscience-data/roots_vi_binhvq_news_corpus)                         |       1 | vi                                                 | News                                                                                                               |
| 80 | [bigscience-data/roots_vi_vietnamese_poetry](https://huggingface.co/datasets/bigscience-data/roots_vi_vietnamese_poetry)                           |       1 | vi                                                 | Poetry                                                                                                             |
| 81 | [DavidLanz/medical_pretrain](https://huggingface.co/datasets/DavidLanz/medical_pretrain)                                                           |       1 | zh                                                 | Biomedical                                                                                                         |
| 82 | [Orion-zhen/dpo-emoji-zh](https://huggingface.co/datasets/Orion-zhen/dpo-emoji-zh)                                                                 |       1 | zh                                                 | Emoji Heavy                                                                                                        |
| 83 | [Orion-zhen/dpo-ruozhiba-emoji](https://huggingface.co/datasets/Orion-zhen/dpo-ruozhiba-emoji)                                                     |       1 | zh                                                 | Emoji Heavy                                                                                                        |
| 84 | [shareAI/DPO-zh-en-emoji](https://huggingface.co/datasets/shareAI/DPO-zh-en-emoji)                                                                 |       1 | zh                                                 | Emoji Heavy                                                                                                        |
| 85 | [Duxiaoman-DI/FinCorpus](https://huggingface.co/datasets/Duxiaoman-DI/FinCorpus)                                                                   |       1 | zh                                                 | Finance                                                                                                            |
| 86 | [TigerResearch/tigerbot-law-plugin](https://huggingface.co/datasets/TigerResearch/tigerbot-law-plugin)                                             |       1 | zh                                                 | Legal                                                                                                              |
| 87 | [bigscience-data/roots_zh_ted_talks_iwslt](https://huggingface.co/datasets/bigscience-data/roots_zh_ted_talks_iwslt)                               |       1 | zh                                                 | Live Speech                                                                                                        |
| 88 | [dirtycomputer/chinese_lyrics](https://huggingface.co/datasets/dirtycomputer/chinese_lyrics)                                                       |       1 | zh                                                 | Lyrics                                                                                                             |
| 89 | [AIR-Bench/qa_news_zh](https://huggingface.co/datasets/AIR-Bench/qa_news_zh)                                                                       |       1 | zh                                                 | News                                                                                                               |
| 90 | [liswei/news-collection-zhtw](https://huggingface.co/datasets/liswei/news-collection-zhtw)                                                         |       1 | zh                                                 | News                                                                                                               |
| 91 | [erhwenkuo/poetry-chinese-zhtw](https://huggingface.co/datasets/erhwenkuo/poetry-chinese-zhtw)                                                     |       1 | zh                                                 | Poetry                                                                                                             |
| 92 | [bigscience-data/roots_zh-cn_wikipedia](https://huggingface.co/datasets/bigscience-data/roots_zh-cn_wikipedia)                                     |       1 | zh                                                 | Wikipedia                                                                                                          |
