# About the African Tokenizer Leaderboard

A tokenizer decides how text is cut into the pieces a language or speech model reads. Most
tokenizers were built mostly from English and a few other large languages, so African
languages are often cut into many more pieces. That makes models slower and more expensive on
them, shortens how much text fits in a context window, and can make them harder to learn.

This leaderboard measures that gap for **313 African languages** and five high-resource
languages widely used in Africa (English, French, Arabic, Portuguese, Spanish), so you can pick
a tokenizer for your language, or see where new ones are needed.

## Metrics

**Tokens per word** (lower is better). Total tokens divided by total words. A value of 1.0 means
whole words are single tokens; 3.0 means an average word is cut into three pieces. Words are
counted by splitting on whitespace, the same way for every language.

**Parity vs English** (lower is better). The number of tokens for a set of Bible verses in the
language, divided by the number of tokens for the *same verses* in English, with the same
tokenizer ([Petrov et al., 2023](https://arxiv.org/abs/2305.15425)). 1.0 means the language
costs as much as English; 2.0 means twice as much. Each language is compared on up to 2,000
verses that exist in both its Bible and the English one.

**Bytes per token** (higher is better). UTF-8 bytes of text divided by tokens: how much text
each token carries.

**Vocabulary size.** How many distinct tokens the tokenizer has. Larger vocabularies tend to
give fewer tokens per word, at the cost of a bigger embedding table in the model.

## Companion languages

Almost every African language is written and spoken alongside a high-resource language:
English, French, Arabic, Portuguese or Spanish. A tokenizer is only really useful for Twi if it
also handles the English around it, so by default:

- **All languages** includes those five high-resource languages along with the 313 African
  ones.
- **Each African language you pick is ranked together with its companion**, the high-resource
  language it is mainly used alongside: Twi with English, Wolof with French, Kimbundu with
  Portuguese, Moroccan Arabic with Arabic. The table shows a column for each.

Companions come from the countries each language is used in (Africa Corpus metadata),
weighted by population, mapped to each country's dominant high-resource language, with
reviewed corrections; the full list is in `benchmark/companion_languages.csv`. Untick
**Include high-resource languages** to score African languages alone.

Several languages are averaged with each language counting equally, so large languages don't
dominate.

## The benchmark text

Built from the [AfriSpeech Africa Corpus](https://huggingface.co/datasets/AfriSpeech/africa-corpus)
with the Africa Corpus Builder. Every language with at least 10,000 distinct Bible sentences is
included. Each language's text combines whichever of these sources it has, up to about 200,000
words from each, and every score uses all of it together:

- AfriSpeech/africa-corpus: Bible text, one translation per language (all languages)
- AfriSpeech/african-corpus-jw: Watchtower articles (where available)
- AfriSpeech/madlad-91-african: web text, clean split; for Twi, ghanaopenai/pristine-twi
  (where available)

Much of this text is religious and formal, so absolute numbers on everyday text can differ;
comparisons between tokenizers on the same text are what the leaderboard is for.
Where a language code bundles several scripts, the main script is used (for example Latin
rather than Ajami for Hausa). Arabic uses the least vowel-marked Bible translation available.

The benchmark text itself is not published, because most of it is publisher-copyrighted;
only the scores are.

## Submitting a tokenizer

Any tokenizer on the Hugging Face Hub that loads with `AutoTokenizer` can be submitted on the
*Submit* page. It is evaluated on every language and added to the table. Repositories that need
`trust_remote_code` are not run.

## Credits

Built by [AfriSpeech](https://huggingface.co/AfriSpeech) on the
[Multilingual Tokenizer Leaderboard](https://huggingface.co/spaces/eduagarcia/multilingual-tokenizer-leaderboard)
by Eduardo A. S. Garcia (Apache-2.0), whose evaluation code computes every metric here.
