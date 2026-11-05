import os
import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["HF_HOME"] = os.path.abspath("data/tokenizers")

model_lang_eval_mapping = {
    "multilingual": None,
    "code": None,
    "en": None,
    "pt": None,#"code, en, pt, es, fr, it, de",
    "es": None,#"code, en, pt, es, fr, it, de",
    "fr": None,#"code, en, pt, es, fr, it, de",
    "it": None,#"code, en, pt, es, fr, it, de",
    "de": None,#"code, en, pt, de",
    "ar": None,#"code, en, pt, ar",
    "ru": None,#"code, en, pt, ru, pl",
    "pl": None,#"code, en, pt, ru, pl",
    "ko": None,#"code, en, pt, ko, zh, ja",
    "zh": None,#"code, en, pt, ko, zh, ja",
    "ja": None,#"code, en, pt, ko, zh, ja",
}

multilingual_models = [
    #bert
    "google-bert/bert-base-multilingual-cased",
    "google-bert/bert-base-multilingual-uncased",
    "google/mt5-base",
    #xlm roberta
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-mlm-17-1280",
    "FacebookAI/xlm-mlm-enfr-1024",
    "FacebookAI/xlm-mlm-xnli15-1024",
    "FacebookAI/xlm-mlm-ende-1024",
    "FacebookAI/xlm-mlm-100-1280",
    "FacebookAI/xlm-mlm-enro-1024",
    "FacebookAI/xlm-mlm-en-2048",
    #microsft
    "microsoft/mdeberta-v3-base",
    #cOHERE
    "CohereLabs/c4ai-command-r-plus",
    "CohereLabs/aya-101",
    #allenai
    "allenai/Molmo-7B-O-0924",
    "allenai/olmOCR-7B-0225-preview",
    "allenai/MolmoE-1B-0924",
    "allenai/tulu-2-dpo-70b",
    #bigscience
    "bigscience/bloom",
    "bigscience/T0pp",
    "bigscience/mt0-xxl",
    #THUDM
    "THUDM/LongWriter-glm4-9b",
    "THUDM/glm-4-9b-chat",
    "THUDM/GLM-4-32B-0414",
    "THUDM/chatglm3-6b-base",
    "THUDM/chatglm3-6b",
    "THUDM/chatglm2-6b",
    #internlm
    "internlm/internlm2-20b-reward",
    "internlm/internlm-xcomposer2d5-7b",
    "internlm/internlm3-8b-instruct",
    "internlm/OREAL-32B",
    "internlm/internlm-chat-20b",
    "internlm/internlm-xcomposer-7b",
    "internlm/Agent-FLAN-7b",
    "internlm/AlchemistCoder-DS-6.7B",
    "internlm/internlm2_5-20b",
    #mistralai
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Nemo-Instruct-2407",
    #01-ai
    "01-ai/Yi-6B",
    "01-ai/Yi-1.5-34B-Chat",
    "01-ai/Yi-34B",
    #Xenova tokenizers
    "Xenova/text-embedding-ada-002",
    "Xenova/gpt-3",
    "Xenova/gpt-3.5-turbo",
    "Xenova/gpt-3.5-turbo-16k",
    "Xenova/gpt-4",
    "Xenova/gpt-4o",
    "Xenova/claude-tokenizer",
    "Xenova/grok-1-tokenizer",
    #freedom intelligence
    "FreedomIntelligence/HuatuoGPT-o1-8B",
    "FreedomIntelligence/HuatuoGPT-o1-7B",
    "FreedomIntelligence/HuatuoGPT-Vision-34B",
    "FreedomIntelligence/DiagnosisGPT-34B",
    "FreedomIntelligence/Apollo2-9B",
    "FreedomIntelligence/phoenix-inst-chat-7b",
    "FreedomIntelligence/HuatuoGPT-13b-delta",
    "FreedomIntelligence/HuatuoGPT2-7B",
    "FreedomIntelligence/ALLaVA-3B",
    "FreedomIntelligence/Apollo-1.8B",
    "FreedomIntelligence/Apollo-7B",
    "FreedomIntelligence/AceGPT-v1.5-13B-Chat",
    "FreedomIntelligence/LongLLaVA-53B-A13B",
    "FreedomIntelligence/ALLaVA-StableLM2-1_6B",
    #others
    "Twitter/twhin-bert-base",
    "papercup-ai/multilingual-pl-bert",
    "Alibaba-NLP/gte-multilingual-base",
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-gemma",
    "BAAI/bge-reranker-v2-minicpm-layerwise",
    "BAAI/bge-reranker-v2.5-gemma2-lightweight",
    "BAAI/bge-multilingual-gemma2",
    "BAAI/BGE-VL-base",
    "intfloat/multilingual-e5-large",
    "almanach/camembertav2-base",
    "Lajavaness/bilingual-embedding-large",
    "OrdalieTech/Solon-embeddings-large-0.1",
    "jinaai/jina-embeddings-v3",
    "jinaai/ReaderLM-v2",
    "jinaai/jina-clip-v1",
    "jinaai/jina-embeddings-v2-base-zh",
    "jinaai/jina-embeddings-v2-base-de",
    "jinaai/jina-embeddings-v2-base-es",
    "studio-ousia/mluke-base",
    "jhu-clsp/bernice",
    "PleIAs/Pleias-RAG-350M",
    "DAMO-NLP-MT/polylm-13b",
    #utter_project
    "utter-project/EuroLLM-9B-Instruct",
    "utter-project/TowerSpeech",
    "utter-project/SpireBase",
    #BSC-LT
    "BSC-LT/RoBERTalex",
    "BSC-LT/roberta-base-bne",
    "BSC-LT/roberta-base-biomedical-clinical-es",
    "BSC-LT/roberta-base-biomedical-es",
    "BSC-LT/roberta-base-ca",
    "BSC-LT/roberta-large-bne",
    "BSC-LT/sciroshot",
    "BSC-LT/mdebertaCA",
    "BSC-LT/NextProcurement_pdfutils",
    "BSC-LT/ALIA-40b",
    "BSC-LT/Flor-6.3B-Instruct",
    "BSC-LT/mRoBERTa",
    "BSC-LT/RoBERTa-ca",
    "BSC-LT/mRoBERTa",
    #ClassCat gpt2/roberta-variants
    "ClassCat/gpt2-base-japanese-v2",
    "ClassCat/roberta-base-spanish",
    "ClassCat/gpt2-base-spanish",
    "ClassCat/roberta-base-catalan",
    "ClassCat/roberta-base-latin-v2",
    "ClassCat/roberta-base-french",
    "ClassCat/gpt2-base-french",
    "ClassCat/gpt2-small-catalan-v2",
    "ClassCat/roberta-small-basque",
    "ClassCat/gpt2-small-basque-v2",
    "ClassCat/roberta-small-greek",
    "ClassCat/gpt2-small-greek-v2",
    #tigerresearch
    "TigerResearch/tigerbot-180b-research",
    "TigerResearch/tigerbot-70b-chat-v2",
    "TigerResearch/tigerbot-70b-base-v2",
    "TigerResearch/MedLink-en",
    #dbmdz
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/bert-large-cased-finetuned-conll03-english",
    "dbmdz/bert-base-finnish-europeana-cased",
    "dbmdz/bert-base-french-europeana-cased",
    "dbmdz/bert-base-historic-dutch-cased",
    "dbmdz/bert-base-historic-english-cased",
    "dbmdz/bert-base-historic-multilingual-cased",
    "dbmdz/bert-base-multilingual-cased-finetuned-conll03-spanish",
    "dbmdz/bert-base-swedish-europeana-cased",
    "dbmdz/bert-base-turkish-128k-cased",
    "dbmdz/bert-base-turkish-128k-uncased",
    "dbmdz/bert-base-turkish-uncased",
    "dbmdz/electra-large-discriminator-finetuned-conll03-english",
    "dbmdz/t5-base-conll03-english",
    "dbmdz/bert-base-historic-multilingual-64k-td-cased",
    #joelniklaus
    "joelniklaus/bert-base-uncased-sem_eval_2010_task_8",
    "joelniklaus/distilbert-based-german-cased-ler",
    "joelniklaus/legal-german-roberta-large",
    "joelniklaus/legal-french-roberta-base",
    "joelniklaus/legal-italian-roberta-base",
    "joelniklaus/legal-swiss-longformer-base",
    "joelniklaus/legal-spanish-roberta-large",
    "joelniklaus/legal-xlm-roberta-large",
    "joelniklaus/legal-bulgarian-roberta-base",
    "joelniklaus/legal-croatian-roberta-base",
    "joelniklaus/legal-czech-roberta-base",
    "joelniklaus/legal-danish-roberta-base",
    "joelniklaus/legal-dutch-roberta-base",
    "joelniklaus/legal-estonian-roberta-base",
    "joelniklaus/legal-finnish-roberta-base",
    "joelniklaus/legal-greek-roberta-base",
    "joelniklaus/legal-hungarian-roberta-base",
    "joelniklaus/legal-irish-roberta-base",
    "joelniklaus/legal-latvian-roberta-base",
    "joelniklaus/legal-lithuanian-roberta-base",
    "joelniklaus/legal-maltese-roberta-base",
    "joelniklaus/legal-polish-roberta-base",
    "joelniklaus/legal-romanian-roberta-base",
    "joelniklaus/legal-slovak-roberta-base",
    "joelniklaus/legal-slovenian-roberta-base",
    "joelniklaus/legal-swedish-roberta-base",
    "joelniklaus/legal-english-roberta-large",
    "joelniklaus/legal-portuguese-roberta-large",
    #sambanova
    "sambanovasystems/BLOOMChat-176B-v1",
    "sambanovasystems/codegen-16B-mono-toolbench",
    "sambanovasystems/starcoder-toolbench",
    "sambanovasystems/SambaCoder-nsql-llama-2-70b",
    "sambanovasystems/SN-13B-8k-Instruct",
    "sambanovasystems/SambaLingo-Arabic-Chat",
    "sambanovasystems/SambaLingo-Bulgarian-Chat",
    "sambanovasystems/SambaLingo-Hungarian-Chat",
    "sambanovasystems/SambaLingo-Japanese-Chat",
    "sambanovasystems/SambaLingo-Russian-Chat",
    "sambanovasystems/SambaLingo-Slovenian-Chat",
    "sambanovasystems/SambaLingo-Serbian-Chat",
    "sambanovasystems/SambaLingo-Thai-Chat",
    "sambanovasystems/SambaLingo-Turkish-Chat",
    #Geotrend models
    "Geotrend/bert-base-10lang-cased",
    "Geotrend/bert-base-15lang-cased",
    "Geotrend/bert-base-25lang-cased",
    "Geotrend/bert-base-en-fr-da-ja-vi-cased",
    "Geotrend/bert-base-en-fr-de-no-da-cased",
    "Geotrend/bert-base-en-fr-es-de-zh-cased",
    "Geotrend/bert-base-en-fr-es-pt-it-cased",
    "Geotrend/bert-base-en-fr-lt-no-pl-cased",
    "Geotrend/bert-base-en-fr-nl-ru-ar-cased",
    "Geotrend/bert-base-en-fr-uk-el-ro-cased",
    "Geotrend/bert-base-en-fr-zh-ja-vi-cased",
    "Geotrend/bert-base-en-cased",
    "Geotrend/bert-base-pt-cased",
    "Geotrend/bert-base-es-cased",
    "Geotrend/bert-base-fr-cased",
    "Geotrend/bert-base-zh-cased",
    "Geotrend/bert-base-ja-cased",
    "Geotrend/bert-base-de-cased",
    "Geotrend/bert-base-ru-cased",
    "Geotrend/bert-base-ar-cased",
    "Geotrend/bert-base-it-cased",
    "Geotrend/bert-base-pl-cased",
    #HELSINK NLP
    "Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU",
    "Helsinki-NLP/opus-mt-ROMANCE-en",
    "Helsinki-NLP/opus-mt-SCANDINAVIA-SCANDINAVIA",
    "Helsinki-NLP/opus-mt-en_el_es_fi-en_el_es_fi",
    "Helsinki-NLP/opus-mt-fi_nb_no_nn_ru_sv_en-SAMI",
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Helsinki-NLP/opus-mt-en-es",
    "Helsinki-NLP/opus-mt-tc-big-en-es",
    "Helsinki-NLP/opus-mt-en-fr",
    "Helsinki-NLP/opus-mt-tc-big-en-fr",
    "Helsinki-NLP/opus-mt-zh-en",
    "Helsinki-NLP/opus-mt-en-zh",
    "Helsinki-NLP/opus-mt-en-jap",
    "Helsinki-NLP/opus-mt-ja-en",
    "Helsinki-NLP/opus-tatoeba-en-ja",
    "Helsinki-NLP/opus-mt-de-en",
    "Helsinki-NLP/opus-mt-en-ru",
    "Helsinki-NLP/opus-mt-en-run",
    "Helsinki-NLP/opus-mt-en-ar",
    "Helsinki-NLP/opus-mt-tc-big-en-ar",
    "Helsinki-NLP/opus-mt-ar-en",
    "Helsinki-NLP/opus-mt-war-en",
    "Helsinki-NLP/opus-mt-tc-big-ar-en",
    "Helsinki-NLP/opus-mt-en-it",
    "Helsinki-NLP/opus-mt-en-itc",
    "Helsinki-NLP/opus-mt-it-en",
    "Helsinki-NLP/opus-mt-tc-big-en-it",
    "Helsinki-NLP/opus-mt-pl-en",
    "Helsinki-NLP/opus-mt-ko-en",
    "Helsinki-NLP/opus-mt-tc-big-en-ko",
    "Helsinki-NLP/opus-mt-tc-big-ko-en",
    #HLPT models
    "HPLT/hplt_bert_base_en",
    "HPLT/hplt_bert_base_2_0_eng-Latn",
    "HPLT/hplt_bert_base_pt",
    "HPLT/hplt_bert_base_2_0_por-Latn",
    "HPLT/hplt_bert_base_es",
    "HPLT/hplt_bert_base_2_0_spa-Latn",
    "HPLT/hplt_bert_base_fr",
    "HPLT/hplt_bert_base_2_0_fra-Latn",
    "HPLT/hplt_bert_base_zh",
    "HPLT/hplt_bert_base_ja",
    "HPLT/hplt_bert_base_2_0_jpn-Jpan",
    "HPLT/hplt_bert_base_de",
    "HPLT/hplt_bert_base_2_0_deu-Latn",
    "HPLT/hplt_bert_base_ru",
    "HPLT/hplt_bert_base_2_0_rus-Cyrl",
    "HPLT/hplt_bert_base_ar",
    "HPLT/hplt_bert_base_it",
    "HPLT/hplt_bert_base_2_0_ita-Latn",
    "HPLT/hplt_bert_base_pl",
    "HPLT/hplt_bert_base_2_0_pol-Latn",
    "HPLT/hplt_bert_base_ko",
    "HPLT/hplt_bert_base_2_0_kor-Hang",

]

english_models = [
    #google
    "google-bert/bert-base-uncased",
    "google-bert/bert-base-cased",
    "google/electra-base-discriminator",
    "albert/albert-base-v2",
    "google-t5/t5-base",
    "google/gemma-7b",
    "google/gemma-2-27b",
    "google/gemma-3-27b-it",
    #facebook
    "FacebookAI/xlm-roberta-large",
    "facebook/roscoe-512-roberta-base",
    "facebook/opt-125m",
    "openai-community/openai-gpt",
    "openai-community/gpt2",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/deepseek-llm-67b-base",
    "apple/DCLM-7B",
    #microsoft
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/phi-1_5",
    "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-4-reasoning",
    "microsoft/Phi-4-multimodal-instruct",
    "microsoft/phi-4",
    "microsoft/Phi-4-mini-instruct",
    "microsoft/bitnet-b1.58-2B-4T",
    #deberta
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-base",
    #Llama
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-Prompt-Guard-2-86M",
    "meta-llama/Prompt-Guard-86M",
    "meta-llama/Llama-Prompt-Guard-2-22M",
    "openlm-research/open_llama_3b_v2",
    "openlm-research/open_llama_13b",
    #tiiuae
    "tiiuae/Falcon-E-3B-Instruct",
    "tiiuae/falcon-40b",
    "tiiuae/Falcon3-10B-Instruct",
    "tiiuae/falcon-rw-1b",
    "tiiuae/falcon-11B",
    "tiiuae/viscon-contextual-captioner",
    #elutherai
    "EleutherAI/gpt-neox-20b",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/TinyStories-restricted",
    "EleutherAI/SimpleStories-restricted",
    "EleutherAI/FineWeb-restricted",
    #huggingfacetb
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/cosmo-1b",
    "HuggingFaceTB/python-edu-scorer",
    "HuggingFaceTB/stack-edu-classifier-python",
    "HuggingFaceTB/finemath-classifier",
    "HuggingFaceTB/finemath-ablation-finemath-infimath-4plus",
    #huggingfacem4
    "HuggingFaceM4/idefics2-8b",
    "HuggingFaceM4/idefics-80b-instruct",
    "HuggingFaceM4/tiny-random-OPTForCausalLM-mismatched-vocab-embed-lengths",
    "HuggingFaceM4/tiny-random-idefics",
    "HuggingFaceM4/Florence-2-DocVQA",
    #nvidia
    "nvidia/Nemotron-Mini-4B-Instruct",
    "nvidia/Hymba-1.5B-Instruct",
    "nvidia/Nemotron-H-8B-Base-8K",
    #llava
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-v1.6-34b-hf",
    "llava-hf/bakLlava-v1-hf",
    #upstage
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "upstage/solar-pro-preview-instruct",
    "upstage/solar-1-mini-tokenizer",
    #ai21labs
    "ai21labs/Jamba-v0.1",
    "ai21labs/AI21-Jamba-Mini-1.5",
    #cerebras
    "cerebras/Cerebras-GPT-13B",
    "cerebras/Cerebras-LLaVA-13B",
    "cerebras/Llama3-DocChat-1.0-8B",
    "cerebras/Dragon-DocChat-Context-Encoder",
    #embed
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Alibaba-NLP/gte-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-en-icl",
    "intfloat/e5-large-v2",
    "jinaai/jina-embeddings-v2-base-en",
    "jinaai/jina-embedding-s-en-v1",
    "jinaai/jina-embedding-t-en-v1",
    #other
    "state-spaces/mamba-2.8b-hf",
    "xlnet/xlnet-base-cased",
    "answerdotai/ModernBERT-base",
    "medicalai/ClinicalBERT",
    "MiniMaxAI/MiniMax-Text-01",
    "MiniMaxAI/MiniMax-VL-01",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "neavo/modern_bert_multilingual",
    "EuroBERT/EuroBERT-210m",
    "chandar-lab/NeoBERT",
    "mosaicml/mpt-7b",
    "databricks/dbrx-base",
    "BioMistral/BioMistral-7B",
    "kyutai/helium-1-2b",
    "kyutai/helium-1-preview-2b",
    "KoichiYasuoka/xlm-roberta-base-english-upos", # 2

]

lang_specific_models = {
    "multilingual": multilingual_models,
    "en": english_models,
    "code": [
        "meta-llama/CodeLlama-7b-hf",
        "meta-llama/CodeLlama-70b-hf",
        "meta-llama/CodeLlama-70b-Instruct-hf",
        "deepseek-ai/deepseek-coder-33b-base",
        "microsoft/codebert-base",
        "huggingface/CodeBERTa-small-v1",
        "EleutherAI/llemma_34b",
        "EleutherAI/llemma_7b",
        "bigcode/starcoder2-15b",
        "bigcode/santacoder",
        "bigcode/starcoder",
        "bigcode/tokenizer-the-stack-march-sample",
        "bigcode/tokenizer-the-stack-march-sample-v2",
        "bigcode/tokenizer-the-stack-march-sample-v3",
        "nvidia/OpenCodeReasoning-Nemotron-32B",
        "JetBrains/Mellum-4b-base",
        "BAAI/bge-code-v1",
        "jinaai/starcoder-1b-textbook",
        "jinaai/jina-embeddings-v2-base-code",
        "replit/replit-code-v1_5-3b",
        "replit/replit-code-v1-3b",
        "HuggingFaceH4/starchat-alpha",
        "HuggingFaceH4/starchat-beta",
        "CAUKiel/JavaBERT",
        "CAUKiel/JavaBERT-uncased",
        "KoichiYasuoka/roberta-large-english-upos", # 8
    ],
    "pt": [
        "eduagarcia/RoBERTaLexPT-base",
        "eduagarcia/RoBERTaCrawlPT-base",
        "pierreguillou/gpt2-small-portuguese",
        "neuralmind/bert-base-portuguese-cased",
        "neuralmind/bert-large-portuguese-cased",
        "maritaca-ai/sabia-2-tokenizer-small",
        "maritaca-ai/sabia-2-tokenizer-medium",
        "maritaca-ai/sabia-7b",
        "unicamp-dl/ptt5-base-portuguese-vocab",
        "unicamp-dl/InRanker-small",
        "unicamp-dl/mMiniLM-L6-v2-mmarco-v2",
        "unicamp-dl/mt5-13b-mmarco-100k",
        "PORTULAN/albertina-100m-portuguese-ptbr-encoder",
        "PORTULAN/albertina-900m-portuguese-ptbr-encoder-brwac",
        "PORTULAN/gervasio-7b-portuguese-ptbr-decoder",
        "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir",
        "nicholasKluge/RewardModel",
        "nicholasKluge/ToxicityModelPT",
        "nicholasKluge/ToxicityModel",
        "nicholasKluge/Aira-2-774M",
        "nicholasKluge/Aira-2-portuguese-560M",
        "nicholasKluge/Aira-2-portuguese-124M",
        "nicholasKluge/Aira-2-1B1",
        "nicholasKluge/Aira-OPT-125M",
        "TucanoBR/Tucano-2b4",
        "dominguesm/legal-bert-base-cased-ptbr",
        "dominguesm/whisper-tiny-pt",
        "dominguesm/mambarim-110m",
        "dominguesm/bert-restore-punctuation-ptbr",
        "dominguesm/positive-reframing-ptbr",
        "dominguesm/positive-reframing-en",
        "dominguesm/canarim-7b",
        "dominguesm/tiny-random-canarim",
        "dominguesm/canarim-bert-nheengatu",
        "cnmoro/Qwen3-0.6B-Portuguese-Tokenizer",
        "melll-uff/bertweetbr",
        "melll-uff/sbert_ptbr",
        "pucpr/biobertpt-all",
        "pucpr/gpt2-bio-pt",
        "pucpr-br/postagger-bio-english",
    ],
    "de": [
        "stefan-it/xlstm-german-wikipedia",
        "stefan-it/electra-base-gc4-64k-0-cased-discriminator",
        "dbmdz/german-gpt2",
        "dbmdz/bert-base-german-cased",
        "stefan-it/zeitungs-lm-v1",
        "Tanhim/gpt2-model-de",
        "benjamin/gerpt2-large",
        "benjamin/gpt2-wechsel-german",
        "benjamin/roberta-base-wechsel-german",
        "benjamin/compoundpiece",
        "dbmdz/distilbert-base-german-europeana-cased",
        "dbmdz/bert-base-german-europeana-uncased",
        "dbmdz/bert-base-german-uncased",
        "google-bert/bert-base-multilingual-cased",
        "google-bert/bert-base-german-cased",
        "google-bert/bert-base-multilingual-uncased",
        "Tanhim/translation-En2De",
        "SebastianBodza/DElefant",
        "SebastianBodza/Kartoffel-1B-v0.3",
        "SebastianBodza/SmolKartoffel-135M-v0.1",
    ],
    "ar": [
        "riotu-lab/ArabianGPT-03B",
        "riotu-lab/Aranizer-SP-64k",
        "riotu-lab/Aranizer-SP-32k",
        "riotu-lab/mushkil",
        "riotu-lab/Aranizer-SP-86k",
        "riotu-lab/Aranizer-PBE-64k",
        "MohamedRashad/arabic-large-nougat",
        "riotu-lab/Aranizer-PBE-32k",
        "asafaya/albert-large-arabic",
        "asafaya/bert-base-arabic",
        "asafaya/kanarya-2b",
        "inceptionai/jais-13b",
        "inceptionai/jais-adapted-70b",
        "MaryamMaksour/arabic-english-tokenizer",
        "MaryamMaksour/arabic-english-tokenizer_1024",
        "MaryamMaksour/arabic-english-tokenizer_2048",
        "kaddu341/dot-arabic-tokenizer1",
        "MohamedRashad/arabic-small-nougat",
        "MohamedRashad/Arabic-Whisper-CodeSwitching-Edition",
        "wissamantoun/araelectra-base-artydiqa",
    ],
    "zh": [
        "google-bert/bert-base-chinese",
        "BAAI/bge-large-zh-v1.5",
        "baichuan-inc/Baichuan-7B",
        "baichuan-inc/Baichuan-Omni-1d5",
        "baichuan-inc/Baichuan-13B-Chat",
        "baichuan-inc/Baichuan2-13B-Chat",
        "baichuan-inc/Baichuan-M1-14B-Instruct",
        "silk-road/luotuo-bert",
        "silk-road/luotuo-bert-medium",
        "clue/xlnet_chinese_large",
        "hfl/chinese-roberta-wwm-ext",
        "hfl/chinese-xlnet-base",
        "hfl/cino-large-v2",
        "hfl/cino-large",
        "hfl/chinese-alpaca-2-7b",
        "hfl/chinese-mixtral-instruct",
        "hfl/llama-3-chinese-8b-instruct-v3",
        #"ckiplab/bert-base-han-chinese-pos-xiandai",
        "Jihuai/bert-ancient-chinese",
        "nghuyong/ernie-3.0-base-zh",
        "nghuyong/ernie-1.0-base-zh",
        "nghuyong/ernie-2.0-base-en",
        "nghuyong/ernie-gram-zh",
        "nghuyong/ernie-health-zh",
        "ethanyt/guwenbert-base",
        "uer/gpt2-chinese-ancient",
        "uer/gpt2-chinese-couplet",
        "uer/gpt2-chinese-poem",
        "uer/roberta-base-word-chinese-cluecorpussmall",
        "uer/t5-base-chinese-cluecorpussmall",
        "uer/t5-small-chinese-cluecorpussmall",
        "CofeAI/FLM-101B",
        "CofeAI/Tele-FLM-1T",
        "KoichiYasuoka/roberta-classical-chinese-base-char", # 9
        "KoichiYasuoka/chinese-bert-wwm-ext-upos", # 8
        "KoichiYasuoka/roberta-base-chinese-upos", # 2
        "KoichiYasuoka/bert-ancient-chinese-base-upos", # 1
        "KoichiYasuoka/deberta-xlarge-chinese-erlangshen-upos", # 1
    ],
    "ko": [
        "EleutherAI/polyglot-ko-1.3b",
        "madatnlp/km-bert",
        "monologg/koelectra-base-v3-discriminator",
        "monologg/kobert",
        "monologg/kobigbird-bert-base",
        "monologg/koelectra-base-discriminator",
        "monologg/koelectra-base-v2-discriminator",
        "klue/roberta-large",
        "lassl/bert-ko-base",
        "lassl/roberta-ko-small",
        "kykim/bert-kor-base",
        "kykim/funnel-kor-base",
        "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "snunlp/KR-BERT-char16424",
        "snunlp/KR-ELECTRA-discriminator",
        "snunlp/KR-FinBert-SC",
        "BM-K/KoChatBART",
        "BM-K/NewsKoT5-small",
        "beomi/EXAONE-3.5-2.4B-Instruct-Llamafied",
        "kakaobank/kf-deberta-base",
        "beomi/KcELECTRA-base-v2022",
        "beomi/KoAlpaca-Polyglot-5.8B",
        "beomi/KcELECTRA-base",
        "beomi/kcbert-base",
        "beomi/KoAlpaca-llama-1-7b",
        "beomi/kollama-13b",
        "beomi/llama-2-ko-7b",
        "beomi/Llama-2-ko-7b-Chat-q4f16_1",
        "beomi/Yi-Ko-6B",
        "beomi/OPEN-SOLAR-KO-10.7B",
        "beomi/SOLAR-KOEN-10.8B",
        "beomi/gemma-ko-7b",
        "beomi/Llama-3-Open-Ko-8B",
        "beomi/Solar-Ko-Recovery-11B",
        "kfkas/Llama-2-ko-7b-Chat",
        "kfkas/Legal-Llama-2-ko-7b-Chat",
        "skt/kobert-base-v1",
        "skt/kogpt2-base-v2",
        "skt/ko-gpt-trinity-1.2B-v0.5",
        "Dongjin-kr/ko-reranker",
        "KoichiYasuoka/roberta-large-korean-upos", # 4
    ],
    "ja": [
        "abeja/gpt-neox-japanese-2.7b",
        "abeja/gpt2-large-japanese",
        "abeja/Mixtral-8x7B-v0.1-japanese",
        "rinna/qwq-bakeneko-32b",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft",
        "rinna/gemma-2-baku-2b-it",
        "rinna/japanese-gpt-1b",
        "rinna/japanese-gpt2-medium",
        "rinna/japanese-gpt-neox-small",
        "rinna/bilingual-gpt-neox-4b",
        "elyza/ELYZA-japanese-Llama-2-7b",
        "rinna/nekomata-14b-instruction",
        "elyza/Llama-3-ELYZA-JP-8B",
        "tohoku-nlp/bert-base-japanese-char-v2",
        "tohoku-nlp/bert-base-japanese-char",
        "tohoku-nlp/bert-base-japanese-v2",
        "tohoku-nlp/bert-base-japanese-whole-word-masking",
        "tohoku-nlp/bert-base-japanese-v3",
        "tohoku-nlp/bert-base-japanese-char-v3",
        "tohoku-nlp/tohokunlp-bert-500m-sq8192-alpha",
        "hitachi-nlp/roberta-base_last-char_acl2023",
        "ybelkada/japanese-dummy-tokenizer",
        "ken11/albert-base-japanese-v1",
        "ken11/mbart-ja-en",
        "ken11/albert-base-japanese-v1-with-japanese-tokenizer",
        "sonoisa/t5-base-japanese",
        "sonoisa/byt5-small-japanese",
        "sonoisa/t5-base-english-japanese",
        "sonoisa/t5-base-japanese-v1.1",
        "sonoisa/sentence-luke-japanese-base-lite",
        "Geotrend/bert-base-ja-cased",
        "Geotrend/distilbert-base-25lang-cased",
        "hotchpotch/japanese-bge-reranker-v2-m3-v1",
        "hotchpotch/xlm-roberta-japanese-tokenizer",
        "hotchpotch/japanese-reranker-tiny-v2",
        "ALINEAR/albert-japanese-v2",
        "ALINEAR/albert-japanese",
        "Fugaku-LLM/Fugaku-LLM-13B",
        "SakuraLLM/Sakura-13B-LNovel-v0.8",
        "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
        "elyza/ELYZA-japanese-CodeLlama-7b-instruct",
        "elyza/ELYZA-japanese-Llama-2-13b-fast-instruct",
        "webbigdata/ALMA-7B-Ja",
        "studio-ousia/luke-base",
        "studio-ousia/luke-japanese-large",
        "KoichiYasuoka/deberta-large-japanese-aozora", # 5
        "KoichiYasuoka/modernbert-base-japanese-wikipedia", # 5
        "KoichiYasuoka/bert-large-japanese-upos", # 3
        "KoichiYasuoka/deberta-large-japanese-wikipedia", # 3
        "KoichiYasuoka/bert-large-japanese-unidic-luw-upos", # 2
        "KoichiYasuoka/roberta-large-japanese-aozora", # 2
    ],
    "fr":[
        "croissantllm/CroissantLLMChat-v0.1",
        "croissantllm/bloom1b7Chat",
        "jpacifico/Chocolatine-3B-Instruct-DPO-Revised",
        "manu/contrastive_zeroner",
        "OpenLLM-France/Claire-Mistral-7B-0.1",
        "manu/tok_custom_scratch",
        "manu/tok_custom_refitted",
        "manu/tok-5M-scratch",
        "manu/tok-2M-scratch",
        "manu/sentence_mdebertav3_v0",
        "Lajavaness/bilingual-embedding-large",
        "manu/colpali-3b-mix-448-docmatix",
        "jpacifico/Chocolatine-2-14B-Instruct-v2.0.3",
        "almanach/camembert-base",
        "almanach/camembert-large",
        "almanach/camembert-base-wikipedia-4gb",
        "almanach/camemberta-base",
        "almanach/manta-lm-base",
        "almanach/camembertav2-base",
        "flaubert/flaubert_base_cased",
        "flaubert/flaubert_base_uncased",
        "Lajavaness/sentence-camembert-large",
        "Lajavaness/sentence-camembert-base",
        "Lajavaness/wav2vec2-lg-xlsr-fr-speech-emotion-recognition",
        "OpenLLM-France/Claire-7B-0.1",
        "OpenLLM-France/Lucie-7B",
        "OpenLLM-France/Lucie-7B-Instruct-v1.1",
        "dbddv01/gpt2-french-small",
        "jpacifico/French-Alpaca-Llama3-8B-Instruct-v1.0",
        "jpacifico/Chocolatine-14B-Instruct-DPO-v1.3",
        "jpacifico/Chocolatine-2-14B-Instruct-v2.0",
        "benjamin/gpt2-wechsel-french",
    ],
    "ru": [
        "yandex/YandexGPT-5-Lite-8B-pretrain",
        "deepvk/RuModernBERT-base",
        "deepvk/roberta-base",
        "deepvk/bert-base-uncased",
        "deepvk/USER-base",
        "deepvk/USER-bge-m3",
        "deepvk/llava-saiga-8b",
        "deepvk/kazRush-ru-kk",
        "deepvk/plato-9b",
        "ai-forever/FRIDA",
        "ai-forever/ru-en-RoSBERTa",
        "ai-forever/sbert_large_nlu_ru",
        "ai-forever/ruRoberta-large",
        "ai-forever/ruT5-large",
        "ai-forever/rugpt3large_based_on_gpt2",
        "ai-forever/mGPT",
        "ai-forever/ruElectra-small",
        "ai-forever/ruElectra-medium",
        "ai-forever/ruSciBERT",
        "ai-forever/ruGPT-3.5-13B",
        "ai-forever/RuM2M100-1.2B",
        "ai-forever/FRED-T5-large-spell",
        "ai-forever/ruElectra-large",
        "ai-forever/mGPT-1.3B-persian",
        "ai-forever/sage-mt5-large",
        "DeepPavlov/rubert-base-cased",
        "msu-rcc-lair/RuadaptQwen2.5-32B-Instruct",
        "cointegrated/rubert-tiny2",
        "DeepPavlov/rubert-base-cased-conversational",
        "t-bank-ai/ruDialoGPT-medium",
        "cointegrated/LaBSE-en-ru",
        "cointegrated/rubert-tiny",
        "cointegrated/rut5-base-multitask",
        "cointegrated/rut5-small-normalizer",
        "cointegrated/rut5-base-labse-decoder",
        "cointegrated/SONAR_200_text_encoder",
        "DeepPavlov/xlm-roberta-large-en-ru",
        "DeepPavlov/bert-base-bg-cs-pl-ru-cased",
        "DeepPavlov/bert-base-cased-conversational",
        "Gherman/bert-base-NER-Russian",
        "DeepPavlov/distilrubert-tiny-cased-conversational-v1",
        "DeepPavlov/marianmt-tatoeba-enru",
        "DeepPavlov/bart-base-en-persona-chat",
        "DeepPavlov/distilrubert-tiny-cased-conversational-5k",
        "DeepPavlov/rudialogpt3_medium_based_on_gpt2_v2",
        "DeepPavlov/mbart-large-50-ru-persona-chat",
        "DeepPavlov/t5-wikidata5M-with-neighbors",
        "Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition",
        "Aniemore/wavlm-emotion-russian-resd",
        "sergeyzh/rubert-mini-uncased",
        "KoichiYasuoka/roberta-small-belarusian", # 5
        "KoichiYasuoka/roberta-base-ukrainian-upos", # 4
        "KoichiYasuoka/bert-base-russian-upos", # 4
        "KoichiYasuoka/bert-base-slavic-cyrillic-upos", # 3
    ],
    "es": [
        "ITG/DialoGPT-medium-spanish-chitchat",
        "bertin-project/bertin-roberta-base-spanish",
        "Narrativa/byt5-base-tweet-hate-detection",
        "Narrativa/distilroberta-finetuned-stereotype-detection",
        "Narrativa/mT5-base-finetuned-tydiQA-question-generation",
        "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation",
        "mrm8488/spanish-gpt2",
        "Narrativa/t5-base-finetuned-totto-table-to-text",
        "PlanTL-GOB-ES/RoBERTalex",
        "Narrativa/NarbioBART",
        "Narrativaai/bloom-560m-finetuned-totto-table-to-text",
        "Narrativaai/deberta-v3-small-finetuned-hate_speech18",
        "PlanTL-GOB-ES/roberta-large-bne",
        "dccuchile/bert-base-spanish-wwm-cased",
        "PlanTL-GOB-ES/bsc-bio-ehr-es",
        "IIC/mdeberta-v3-base-cantemist",
        "IIC/mdeberta-v3-base-ehealth_kd",
        "IIC/RigoChat-7b-v2",
        "dccuchile/albert-base-spanish",
        "dccuchile/bert-base-spanish-wwm-uncased",
        "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "PlanTL-GOB-ES/roberta-base-biomedical-es",
        "PlanTL-GOB-ES/roberta-base-ca",
        "PlanTL-GOB-ES/bsc-bio-es",
        "clibrain/lince-zero",
        "clibrain/Llama-2-7b-ft-instruct-es",
        "clibrain/lince-mistral-7b-it-es",
        "4i-ai/BERT_disfluency_cls",
        "DeepESP/gpt2-spanish",
        "datificate/gpt2-small-spanish",
        "mrm8488/longformer-base-4096-spanish",
        "bertin-project/bertin-gpt-j-6B",
        "pysentimiento/robertuito-base-cased",
        "pysentimiento/robertuito-base-uncased",
        "pysentimiento/robertuito-sentiment-analysis",
        "pysentimiento/robertuito-pos",
        "mrm8488/GuaPeTe-2-tiny-finetuned-TED",
        "mrm8488/RuPERTa-base",
        "mrm8488/biomedtra-small-finenuned-clinical-ner",
        "mrm8488/ViT2GPT-2-es",
        "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
        "mrm8488/bert2bert-spanish-question-generation",
        "mrm8488/distilbert-base-multi-cased-finetuned-typo-detection",
        "mrm8488/electricidad-small-finetuned-squadv1-es",
        "mrm8488/spanish-t5-small-sqac-for-qa",
        "mrm8488/wav2vec2-large-xlsr-53-spanish",
        "mrm8488/bart-legal-base-es",
    ],
    "it": [
        "dbmdz/bert-base-italian-xxl-cased",
        "indigo-ai/BERTino",
        "dlicari/Italian-Legal-BERT-SC",
        "dlicari/lsg16k-mbart-summarization-fanpage",
        "swap-uniba/LLaMAntino-2-chat-13b-hf-UltraChat-ITA",
        "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA",
        "swap-uniba/bloom-1b7-it",
        "anakin87/gemma-2-9b-neogenesis-ita",
        "nickprock/distilbert-base-uncased-banking77-classification",
        "nickprock/bert-finetuned-ner-ontonotes",
        "nickprock/setfit-banking77",
        "nickprock/stsbm-sentence-flare-it",
        "nickprock/whisper-tiny-minds14-en",
        "nickprock/speecht5_finetuned_voxpopuli_it_example",
        "nickprock/ModernBERT-base-ita-embed-mnrl",
        "nickprock/Italian-ModernBERT-base-embed-mmarco-triplet",
    ],
    "pl": [
        "Remek/Llama-3-8B-Omnibus-1-PL-v01-INSTRUCT",
        "dkleczek/Polish-Hate-Speech-Detection-Herbert-Large",
        "dkleczek/Polish_BART_base_OPI",
        "dkleczek/Polish_RoBERTa_large_OPI",
        "sdadas/polish-roberta-base-v2",
        "dkleczek/bert-base-polish-cased-v1",
        "dkleczek/bert-base-polish-uncased-v1",
        "dkleczek/papuGaPT2-finetuned-wierszyki",
        "sdadas/mmlw-retrieval-roberta-large",
        "sdadas/polish-gpt2-xl",
        "sdadas/mmlw-retrieval-e5-large",
        "sdadas/mt5-base-translator-en-pl",
        "sdadas/byt5-text-correction",
        "sdadas/stella-pl-retrieval",
    ]

}

GEOTREND_MODELS = [
    "Geotrend/bert-base-en-fr-de-cased",
    "Geotrend/bert-base-en-fr-es-cased",
    "Geotrend/bert-base-en-fr-it-cased",
    "Geotrend/bert-base-en-fr-zh-cased",
    "Geotrend/bert-base-ar-cased",
    "Geotrend/bert-base-bg-cased",
    "Geotrend/bert-base-da-cased",
    "Geotrend/bert-base-de-cased",
    "Geotrend/bert-base-el-cased",
    "Geotrend/bert-base-en-ar-cased",
    "Geotrend/bert-base-en-bg-cased",
    "Geotrend/bert-base-en-cased",
    "Geotrend/bert-base-en-da-cased",
    "Geotrend/bert-base-en-de-cased",
    "Geotrend/bert-base-en-el-cased",
    "Geotrend/bert-base-en-el-ru-cased",
    "Geotrend/bert-base-en-es-cased",
    "Geotrend/bert-base-en-es-it-cased",
    "Geotrend/bert-base-en-es-pt-cased",
    "Geotrend/bert-base-en-es-zh-cased",
    "Geotrend/bert-base-en-fr-ar-cased",
    "Geotrend/bert-base-en-fr-cased",
    "Geotrend/bert-base-en-hi-cased",
    "Geotrend/bert-base-en-it-cased",
    "Geotrend/bert-base-en-ja-cased",
    "Geotrend/bert-base-en-lt-cased",
    "Geotrend/bert-base-en-nl-cased",
    "Geotrend/bert-base-en-no-cased",
    "Geotrend/bert-base-en-pl-cased",
    "Geotrend/bert-base-en-pt-cased",
    "Geotrend/bert-base-en-ro-cased",
    "Geotrend/bert-base-en-ru-cased",
    "Geotrend/bert-base-en-sw-cased",
    "Geotrend/bert-base-en-th-cased",
    "Geotrend/bert-base-en-tr-cased",
    "Geotrend/bert-base-en-uk-cased",
    "Geotrend/bert-base-en-ur-cased",
    "Geotrend/bert-base-en-vi-cased",
    "Geotrend/bert-base-en-zh-cased",
    "Geotrend/bert-base-en-zh-hi-cased",
    "Geotrend/bert-base-es-cased",
    "Geotrend/bert-base-fr-cased",
    "Geotrend/bert-base-hi-cased",
    "Geotrend/bert-base-it-cased",
    "Geotrend/bert-base-ja-cased",
    "Geotrend/bert-base-lt-cased",
    "Geotrend/bert-base-nl-cased",
    "Geotrend/bert-base-no-cased",
    "Geotrend/bert-base-pl-cased",
    "Geotrend/bert-base-pt-cased",
    "Geotrend/bert-base-ro-cased",
    "Geotrend/bert-base-ru-cased",
    "Geotrend/bert-base-sw-cased",
    "Geotrend/bert-base-th-cased",
    "Geotrend/bert-base-tr-cased",
    "Geotrend/bert-base-uk-cased",
    "Geotrend/bert-base-ur-cased",
    "Geotrend/bert-base-vi-cased",
    "Geotrend/bert-base-zh-cased",
]
          
HPLT_MODELS = [
    "HPLT/hplt_bert_base_en",
    "HPLT/hplt_bert_base_af",
    "HPLT/hplt_bert_base_ar",
    "HPLT/hplt_bert_base_az",
    "HPLT/hplt_bert_base_be",
    "HPLT/hplt_bert_base_bg",
    "HPLT/hplt_bert_base_bn",
    "HPLT/hplt_bert_base_ca",
    "HPLT/hplt_bert_base_cs",
    "HPLT/hplt_bert_base_cy",
    "HPLT/hplt_bert_base_da",
    "HPLT/hplt_bert_base_de",
    "HPLT/hplt_bert_base_el",
    "HPLT/hplt_bert_base_eo",
    "HPLT/hplt_bert_base_es",
    "HPLT/hplt_bert_base_et",
    "HPLT/hplt_bert_base_eu",
    "HPLT/hplt_bert_base_fa",
    "HPLT/hplt_bert_base_fi",
    "HPLT/hplt_bert_base_fr",
    "HPLT/hplt_bert_base_ga",
    "HPLT/hplt_bert_base_gl",
    "HPLT/hplt_bert_base_hbs",
    "HPLT/hplt_bert_base_he",
    "HPLT/hplt_bert_base_hi",
    "HPLT/hplt_bert_base_hu",
    "HPLT/hplt_bert_base_hy",
    "HPLT/hplt_bert_base_id",
    "HPLT/hplt_bert_base_is",
    "HPLT/hplt_bert_base_it",
    "HPLT/hplt_bert_base_ja",
    "HPLT/hplt_bert_base_ka",
    "HPLT/hplt_bert_base_kk",
    "HPLT/hplt_bert_base_kn",
    "HPLT/hplt_bert_base_ko",
    "HPLT/hplt_bert_base_ky",
    "HPLT/hplt_bert_base_la",
    "HPLT/hplt_bert_base_lt",
    "HPLT/hplt_bert_base_lv",
    "HPLT/hplt_bert_base_mk",
    "HPLT/hplt_bert_base_ml",
    "HPLT/hplt_bert_base_mn",
    "HPLT/hplt_bert_base_mr",
    "HPLT/hplt_bert_base_ms",
    "HPLT/hplt_bert_base_mt",
    "HPLT/hplt_bert_base_my",
    "HPLT/hplt_bert_base_nb",
    "HPLT/hplt_bert_base_ne",
    "HPLT/hplt_bert_base_nl",
    "HPLT/hplt_bert_base_nn",
    "HPLT/hplt_bert_base_pa",
    "HPLT/hplt_bert_base_pl",
    "HPLT/hplt_bert_base_ps",
    "HPLT/hplt_bert_base_pt",
    "HPLT/hplt_bert_base_ro",
    "HPLT/hplt_bert_base_ru",
    "HPLT/hplt_bert_base_si",
    "HPLT/hplt_bert_base_sk",
    "HPLT/hplt_bert_base_sl",
    "HPLT/hplt_bert_base_so",
    "HPLT/hplt_bert_base_sq",
    "HPLT/hplt_bert_base_sv",
    "HPLT/hplt_bert_base_sw",
    "HPLT/hplt_bert_base_ta",
    "HPLT/hplt_bert_base_te",
    "HPLT/hplt_bert_base_th",
    "HPLT/hplt_bert_base_tl",
    "HPLT/hplt_bert_base_tr",
    "HPLT/hplt_bert_base_tt",
    "HPLT/hplt_bert_base_uk",
    "HPLT/hplt_bert_base_ur",
    "HPLT/hplt_bert_base_uz",
    "HPLT/hplt_bert_base_vi",
    "HPLT/hplt_bert_base_zh",
    "HPLT/hplt_bert_base_2_0_eng-Latn",
    "HPLT/hplt_bert_base_2_0_bel-Cyrl",
    "HPLT/hplt_bert_base_2_0_slv-Latn",
    "HPLT/hplt_bert_base_2_0_spa-Latn",
    "HPLT/hplt_bert_base_2_0_srp-Cyrl",
    "HPLT/hplt_bert_base_2_0_swe-Latn",
    "HPLT/hplt_bert_base_2_0_tat-Cyrl",
    "HPLT/hplt_bert_base_2_0_tur-Latn",
    "HPLT/hplt_bert_base_2_0_ukr-Cyrl",
    "HPLT/hplt_bert_base_2_0_vie-Latn",
    "HPLT/hplt_bert_base_2_0_zho-Hans",
    "HPLT/hplt_bert_base_2_0_als-Latn",
    "HPLT/hplt_bert_base_2_0_bos-Latn",
    "HPLT/hplt_bert_base_2_0_bul-Cyrl",
    "HPLT/hplt_bert_base_2_0_cat-Latn",
    "HPLT/hplt_bert_base_2_0_ces-Latn",
    "HPLT/hplt_bert_base_2_0_cym-Latn",
    "HPLT/hplt_bert_base_2_0_dan-Latn",
    "HPLT/hplt_bert_base_2_0_deu-Latn",
    "HPLT/hplt_bert_base_2_0_ell-Grek",
    "HPLT/hplt_bert_base_2_0_est-Latn",
    "HPLT/hplt_bert_base_2_0_eus-Latn",
    "HPLT/hplt_bert_base_2_0_fao-Latn",
    "HPLT/hplt_bert_base_2_0_fin-Latn",
    "HPLT/hplt_bert_base_2_0_fra-Latn",
    "HPLT/hplt_bert_base_2_0_gla-Latn",
    "HPLT/hplt_bert_base_2_0_gle-Latn",
    "HPLT/hplt_bert_base_2_0_glg-Latn",
    "HPLT/hplt_bert_base_2_0_heb-Hebr",
    "HPLT/hplt_bert_base_2_0_hin-Deva",
    "HPLT/hplt_bert_base_2_0_hrv-Latn",
    "HPLT/hplt_bert_base_2_0_hun-Latn",
    "HPLT/hplt_bert_base_2_0_hye-Armn",
    "HPLT/hplt_bert_base_2_0_ind-Latn",
    "HPLT/hplt_bert_base_2_0_isl-Latn",
    "HPLT/hplt_bert_base_2_0_ita-Latn",
    "HPLT/hplt_bert_base_2_0_jpn-Jpan",
    "HPLT/hplt_bert_base_2_0_kat-Geor",
    "HPLT/hplt_bert_base_2_0_kor-Hang",
    "HPLT/hplt_bert_base_2_0_lit-Latn",
    "HPLT/hplt_bert_base_2_0_ltz-Latn",
    "HPLT/hplt_bert_base_2_0_lvs-Latn",
    "HPLT/hplt_bert_base_2_0_mkd-Latn",
    "HPLT/hplt_bert_base_2_0_mlt-Latn",
    "HPLT/hplt_bert_base_2_0_nld-Latn",
    "HPLT/hplt_bert_base_2_0_nno-Latn",
    "HPLT/hplt_bert_base_2_0_nob-Latn",
    "HPLT/hplt_bert_base_2_0_pes-Arab",
    "HPLT/hplt_bert_base_2_0_pol-Latn",
    "HPLT/hplt_bert_base_2_0_por-Latn",
    "HPLT/hplt_bert_base_2_0_ron-Latn",
    "HPLT/hplt_bert_base_2_0_rus-Cyrl",
    "HPLT/hplt_bert_base_2_0_slk-Latn",
    "HPLT/hplt_bert_base_swh-Latn",
]

helsinki_models = [
    "Helsinki-NLP/opus-mt-zh-en",
    "Helsinki-NLP/opus-mt-en-es",
    "Helsinki-NLP/opus-mt-en-fr",
    "Helsinki-NLP/opus-mt-en-trk",
    "Helsinki-NLP/opus-mt-en-zh",
    "Helsinki-NLP/opus-mt-en-ru",
    "Helsinki-NLP/opus-mt-bn-en",
    "Helsinki-NLP/opus-mt-en-ar",
    "Helsinki-NLP/opus-mt-en-hi",
    "Helsinki-NLP/opus-mt-en-iir",
    "Helsinki-NLP/opus-mt-en-jap",
    "Helsinki-NLP/opus-mt-en-uk",
    "Helsinki-NLP/opus-mt-en-vi",
    "Helsinki-NLP/opus-mt-ja-en",
    "Helsinki-NLP/opus-mt-ja-es",
    "Helsinki-NLP/opus-mt-ko-en",
    "Helsinki-NLP/opus-mt-mul-en",
    "Helsinki-NLP/opus-mt-tr-en",
    "Helsinki-NLP/opus-mt-vi-en",
    "Helsinki-NLP/opus-mt-tc-big-en-ar",
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Helsinki-NLP/opus-mt-tc-big-en-es",
    "Helsinki-NLP/opus-mt-tc-big-en-tr",
    "Helsinki-NLP/opus-mt-tc-big-he-en",
    "Helsinki-NLP/opus-mt-tc-big-en-ko",
    "Helsinki-NLP/opus-mt-tc-big-ko-en",
    "Helsinki-NLP/opus-mt-aav-en",
    "Helsinki-NLP/opus-mt-aed-es",
    "Helsinki-NLP/opus-mt-af-de",
    "Helsinki-NLP/opus-mt-af-en",
    "Helsinki-NLP/opus-mt-af-eo",
    "Helsinki-NLP/opus-mt-af-es",
    "Helsinki-NLP/opus-mt-af-fi",
    "Helsinki-NLP/opus-mt-af-fr",
    "Helsinki-NLP/opus-mt-af-nl",
    "Helsinki-NLP/opus-mt-af-ru",
    "Helsinki-NLP/opus-mt-af-sv",
    "Helsinki-NLP/opus-mt-afa-afa",
    "Helsinki-NLP/opus-mt-afa-en",
    "Helsinki-NLP/opus-mt-alv-en",
    "Helsinki-NLP/opus-mt-am-sv",
    "Helsinki-NLP/opus-mt-ar-de",
    "Helsinki-NLP/opus-mt-ar-el",
    "Helsinki-NLP/opus-mt-ar-en",
    "Helsinki-NLP/opus-mt-ar-eo",
    "Helsinki-NLP/opus-mt-ar-es",
    "Helsinki-NLP/opus-mt-ar-fr",
    "Helsinki-NLP/opus-mt-ar-he",
    "Helsinki-NLP/opus-mt-ar-it",
    "Helsinki-NLP/opus-mt-ar-pl",
    "Helsinki-NLP/opus-mt-ar-ru",
    "Helsinki-NLP/opus-mt-ar-tr",
    "Helsinki-NLP/opus-mt-art-en",
    "Helsinki-NLP/opus-mt-ase-de",
    "Helsinki-NLP/opus-mt-ase-en",
    "Helsinki-NLP/opus-mt-ase-es",
    "Helsinki-NLP/opus-mt-ase-fr",
    "Helsinki-NLP/opus-mt-ase-sv",
    "Helsinki-NLP/opus-mt-en-az",
    "Helsinki-NLP/opus-mt-az-es",
    "Helsinki-NLP/opus-mt-az-tr",
    "Helsinki-NLP/opus-mt-bat-en",
    "Helsinki-NLP/opus-mt-bcl-de",
    "Helsinki-NLP/opus-mt-bcl-en",
    "Helsinki-NLP/opus-mt-bcl-es",
    "Helsinki-NLP/opus-mt-bcl-fi",
    "Helsinki-NLP/opus-mt-bcl-fr",
    "Helsinki-NLP/opus-mt-bcl-sv",
    "Helsinki-NLP/opus-mt-be-es",
    "Helsinki-NLP/opus-mt-en-bem",
    "Helsinki-NLP/opus-mt-bem-es",
    "Helsinki-NLP/opus-mt-bem-fi",
    "Helsinki-NLP/opus-mt-bem-fr",
    "Helsinki-NLP/opus-mt-bem-sv",
    "Helsinki-NLP/opus-mt-en-ber",
    "Helsinki-NLP/opus-mt-ber-es",
    "Helsinki-NLP/opus-mt-ber-fr",
    "Helsinki-NLP/opus-mt-bg-de",
    "Helsinki-NLP/opus-mt-bg-en",
    "Helsinki-NLP/opus-mt-bg-eo",
    "Helsinki-NLP/opus-mt-bg-es",
    "Helsinki-NLP/opus-mt-bg-fi",
    "Helsinki-NLP/opus-mt-bg-fr",
    "Helsinki-NLP/opus-mt-bg-it",
    "Helsinki-NLP/opus-mt-bg-ru",
    "Helsinki-NLP/opus-mt-bg-sv",
    "Helsinki-NLP/opus-mt-bg-tr",
    "Helsinki-NLP/opus-mt-bg-uk",
    "Helsinki-NLP/opus-mt-en-bi",
    "Helsinki-NLP/opus-mt-bi-es",
    "Helsinki-NLP/opus-mt-bi-fr",
    "Helsinki-NLP/opus-mt-bi-sv",
    "Helsinki-NLP/opus-mt-bnt-en",
    "Helsinki-NLP/opus-mt-bzs-en",
    "Helsinki-NLP/opus-mt-bzs-es",
    "Helsinki-NLP/opus-mt-bzs-fi",
    "Helsinki-NLP/opus-mt-bzs-fr",
    "Helsinki-NLP/opus-mt-bzs-sv",
    "Helsinki-NLP/opus-mt-ca-de",
    "Helsinki-NLP/opus-mt-en-ca",
    "Helsinki-NLP/opus-mt-ca-es",
    "Helsinki-NLP/opus-mt-ca-fr",
    "Helsinki-NLP/opus-mt-ca-it",
    "Helsinki-NLP/opus-mt-ca-nl",
    "Helsinki-NLP/opus-mt-ca-pt",
    "Helsinki-NLP/opus-mt-ca-uk",
    "Helsinki-NLP/opus-mt-cau-en",
    "Helsinki-NLP/opus-mt-ccs-en",
    "Helsinki-NLP/opus-mt-ceb-en",
    "Helsinki-NLP/opus-mt-ceb-es",
    "Helsinki-NLP/opus-mt-ceb-fi",
    "Helsinki-NLP/opus-mt-ceb-fr",
    "Helsinki-NLP/opus-mt-ceb-sv",
    "Helsinki-NLP/opus-mt-cel-en",
    "Helsinki-NLP/opus-mt-chk-en",
    "Helsinki-NLP/opus-mt-chk-es",
    "Helsinki-NLP/opus-mt-chk-fr",
    "Helsinki-NLP/opus-mt-chk-sv",
    "Helsinki-NLP/opus-mt-cpf-en",
    "Helsinki-NLP/opus-mt-cpp-cpp",
    "Helsinki-NLP/opus-mt-cpp-en",
    "Helsinki-NLP/opus-mt-crs-de",
    "Helsinki-NLP/opus-mt-crs-en",
    "Helsinki-NLP/opus-mt-crs-es",
    "Helsinki-NLP/opus-mt-crs-fi",
    "Helsinki-NLP/opus-mt-crs-fr",
    "Helsinki-NLP/opus-mt-crs-sv",
    "Helsinki-NLP/opus-mt-cs-de",
    "Helsinki-NLP/opus-mt-en-cs",
    "Helsinki-NLP/opus-mt-cs-eo",
    "Helsinki-NLP/opus-mt-cs-fi",
    "Helsinki-NLP/opus-mt-cs-fr",
    "Helsinki-NLP/opus-mt-cs-sv",
    "Helsinki-NLP/opus-mt-cs-uk",
    "Helsinki-NLP/opus-mt-csg-es",
    "Helsinki-NLP/opus-mt-csn-es",
    "Helsinki-NLP/opus-mt-cus-en",
    "Helsinki-NLP/opus-mt-cy-en",
    "Helsinki-NLP/opus-mt-de-da",
    "Helsinki-NLP/opus-mt-da-en",
    "Helsinki-NLP/opus-mt-da-eo",
    "Helsinki-NLP/opus-mt-da-es",
    "Helsinki-NLP/opus-mt-da-fi",
    "Helsinki-NLP/opus-mt-da-fr",
    "Helsinki-NLP/opus-mt-da-no",
    "Helsinki-NLP/opus-mt-da-ru",
    "Helsinki-NLP/opus-mt-de-ZH",
    "Helsinki-NLP/opus-mt-de-af",
    "Helsinki-NLP/opus-mt-de-ar",
    "Helsinki-NLP/opus-mt-de-bi",
    "Helsinki-NLP/opus-mt-de-bzs",
    "Helsinki-NLP/opus-mt-de-de",
    "Helsinki-NLP/opus-mt-de-ee",
    "Helsinki-NLP/opus-mt-de-efi",
    "Helsinki-NLP/opus-mt-de-el",
    "Helsinki-NLP/opus-mt-de-en",
    "Helsinki-NLP/opus-mt-de-eo",
    "Helsinki-NLP/opus-mt-de-es",
    "Helsinki-NLP/opus-mt-de-et",
    "Helsinki-NLP/opus-mt-de-eu",
    "Helsinki-NLP/opus-mt-de-fi",
    "Helsinki-NLP/opus-mt-de-fj",
    "Helsinki-NLP/opus-mt-de-fr",
    "Helsinki-NLP/opus-mt-de-gaa",
    "Helsinki-NLP/opus-mt-de-gil",
    "Helsinki-NLP/opus-mt-de-guw",
    "Helsinki-NLP/opus-mt-de-ha",
    "Helsinki-NLP/opus-mt-he-de",
    "Helsinki-NLP/opus-mt-de-hil",
    "Helsinki-NLP/opus-mt-de-ho",
    "Helsinki-NLP/opus-mt-de-hr",
    "Helsinki-NLP/opus-mt-de-ht",
    "Helsinki-NLP/opus-mt-de-hu",
    "Helsinki-NLP/opus-mt-de-ig",
    "Helsinki-NLP/opus-mt-de-ilo",
    "Helsinki-NLP/opus-mt-de-is",
    "Helsinki-NLP/opus-mt-de-iso",
    "Helsinki-NLP/opus-mt-de-it",
    "Helsinki-NLP/opus-mt-de-kg",
    "Helsinki-NLP/opus-mt-de-ln",
    "Helsinki-NLP/opus-mt-de-loz",
    "Helsinki-NLP/opus-mt-de-lt",
    "Helsinki-NLP/opus-mt-de-lua",
    "Helsinki-NLP/opus-mt-de-ms",
    "Helsinki-NLP/opus-mt-de-mt",
    "Helsinki-NLP/opus-mt-de-niu",
    "Helsinki-NLP/opus-mt-de-nl",
    "Helsinki-NLP/opus-mt-de-no",
    "Helsinki-NLP/opus-mt-de-nso",
    "Helsinki-NLP/opus-mt-de-ny",
    "Helsinki-NLP/opus-mt-de-pag",
    "Helsinki-NLP/opus-mt-de-pap",
    "Helsinki-NLP/opus-mt-de-pis",
    "Helsinki-NLP/opus-mt-de-pl",
    "Helsinki-NLP/opus-mt-de-pon",
    "Helsinki-NLP/opus-mt-de-tl",
    "Helsinki-NLP/opus-mt-de-uk",
    "Helsinki-NLP/opus-mt-de-vi",
    "Helsinki-NLP/opus-mt-dra-en",
    "Helsinki-NLP/opus-mt-ee-en",
    "Helsinki-NLP/opus-mt-ee-es",
    "Helsinki-NLP/opus-mt-ee-fi",
    "Helsinki-NLP/opus-mt-ee-fr",
    "Helsinki-NLP/opus-mt-ee-sv",
    "Helsinki-NLP/opus-mt-efi-en",
    "Helsinki-NLP/opus-mt-efi-fi",
    "Helsinki-NLP/opus-mt-efi-fr",
    "Helsinki-NLP/opus-mt-efi-sv",
    "Helsinki-NLP/opus-mt-el-ar",
    "Helsinki-NLP/opus-mt-el-eo",
    "Helsinki-NLP/opus-mt-el-fi",
    "Helsinki-NLP/opus-mt-el-fr",
    "Helsinki-NLP/opus-mt-el-sv",
    "Helsinki-NLP/opus-mt-en-CELTIC",
    "Helsinki-NLP/opus-mt-en-ROMANCE",
    "Helsinki-NLP/opus-mt-en-aav",
    "Helsinki-NLP/opus-mt-en-afa",
    "Helsinki-NLP/opus-mt-en-alv",
    "Helsinki-NLP/opus-mt-en-bat",
    "Helsinki-NLP/opus-mt-en-bg",
    "Helsinki-NLP/opus-mt-en-bnt",
    "Helsinki-NLP/opus-mt-en-ceb",
    "Helsinki-NLP/opus-mt-en-cel",
    "Helsinki-NLP/opus-mt-en-cpf",
    "Helsinki-NLP/opus-mt-en-cpp",
    "Helsinki-NLP/opus-mt-en-dra",
    "Helsinki-NLP/opus-mt-en-el",
    "Helsinki-NLP/opus-mt-en-eo",
    "Helsinki-NLP/opus-mt-et-en",
    "Helsinki-NLP/opus-mt-en-eu",
    "Helsinki-NLP/opus-mt-en-euq",
    "Helsinki-NLP/opus-mt-en-fi",
    "Helsinki-NLP/opus-mt-en-fiu",
    "Helsinki-NLP/opus-mt-en-fj",
    "Helsinki-NLP/opus-mt-en-ga",
    "Helsinki-NLP/opus-mt-en-gaa",
    "Helsinki-NLP/opus-mt-en-gem",
    "Helsinki-NLP/opus-mt-en-gil",
    "Helsinki-NLP/opus-mt-en-gl",
    "Helsinki-NLP/opus-mt-en-gmq",
    "Helsinki-NLP/opus-mt-en-gmw",
    "Helsinki-NLP/opus-mt-en-grk",
    "Helsinki-NLP/opus-mt-en-guw",
    "Helsinki-NLP/opus-mt-en-gv",
    "Helsinki-NLP/opus-mt-en-ha",
    "Helsinki-NLP/opus-mt-en-he",
    "Helsinki-NLP/opus-mt-en-hil",
    "Helsinki-NLP/opus-mt-en-ho",
    "Helsinki-NLP/opus-mt-en-ht",
    "Helsinki-NLP/opus-mt-en-hu",
    "Helsinki-NLP/opus-mt-en-hy",
    "Helsinki-NLP/opus-mt-en-id",
    "Helsinki-NLP/opus-mt-en-ig",
    "Helsinki-NLP/opus-mt-en-ilo",
    "Helsinki-NLP/opus-mt-en-inc",
    "Helsinki-NLP/opus-mt-en-ine",
    "Helsinki-NLP/opus-mt-is-en",
    "Helsinki-NLP/opus-mt-en-iso",
    "Helsinki-NLP/opus-mt-en-it",
    "Helsinki-NLP/opus-mt-en-itc",
    "Helsinki-NLP/opus-mt-en-kg",
    "Helsinki-NLP/opus-mt-en-kj",
    "Helsinki-NLP/opus-mt-en-kqn",
    "Helsinki-NLP/opus-mt-en-kwn",
    "Helsinki-NLP/opus-mt-en-kwy",
    "Helsinki-NLP/opus-mt-en-lg",
    "Helsinki-NLP/opus-mt-en-ln",
    "Helsinki-NLP/opus-mt-en-loz",
    "Helsinki-NLP/opus-mt-en-lu",
    "Helsinki-NLP/opus-mt-en-lua",
    "Helsinki-NLP/opus-mt-en-lue",
    "Helsinki-NLP/opus-mt-en-lun",
    "Helsinki-NLP/opus-mt-en-luo",
    "Helsinki-NLP/opus-mt-lus-en",
    "Helsinki-NLP/opus-mt-en-map",
    "Helsinki-NLP/opus-mt-en-mfe",
    "Helsinki-NLP/opus-mt-en-mg",
    "Helsinki-NLP/opus-mt-en-mh",
    "Helsinki-NLP/opus-mt-mk-en",
    "Helsinki-NLP/opus-mt-en-mkh",
    "Helsinki-NLP/opus-mt-en-ml",
    "Helsinki-NLP/opus-mt-en-mos",
    "Helsinki-NLP/opus-mt-en-mr",
    "Helsinki-NLP/opus-mt-en-mt",
    "Helsinki-NLP/opus-mt-en-mul",
    "Helsinki-NLP/opus-mt-en-ng",
    "Helsinki-NLP/opus-mt-en-nic",
    "Helsinki-NLP/opus-mt-en-niu",
    "Helsinki-NLP/opus-mt-nl-en",
    "Helsinki-NLP/opus-mt-en-nso",
    "Helsinki-NLP/opus-mt-en-ny",
    "Helsinki-NLP/opus-mt-en-nyk",
    "Helsinki-NLP/opus-mt-en-om",
    "Helsinki-NLP/opus-mt-pag-en",
    "Helsinki-NLP/opus-mt-en-pap",
    "Helsinki-NLP/opus-mt-en-phi",
    "Helsinki-NLP/opus-mt-en-pis",
    "Helsinki-NLP/opus-mt-en-pon",
    "Helsinki-NLP/opus-mt-en-poz",
    "Helsinki-NLP/opus-mt-en-pqe",
    "Helsinki-NLP/opus-mt-en-pqw",
    "Helsinki-NLP/opus-mt-en-rn",
    "Helsinki-NLP/opus-mt-en-rnd",
    "Helsinki-NLP/opus-mt-en-ro",
    "Helsinki-NLP/opus-mt-en-roa",
    "Helsinki-NLP/opus-mt-en-run",
    "Helsinki-NLP/opus-mt-rw-en",
    "Helsinki-NLP/opus-mt-en-sal",
    "Helsinki-NLP/opus-mt-en-sem",
    "Helsinki-NLP/opus-mt-en-sg",
    "Helsinki-NLP/opus-mt-en-sit",
    "Helsinki-NLP/opus-mt-en-sk",
    "Helsinki-NLP/opus-mt-en-sla",
    "Helsinki-NLP/opus-mt-en-sm",
    "Helsinki-NLP/opus-mt-en-sn",
    "Helsinki-NLP/opus-mt-en-sq",
    "Helsinki-NLP/opus-mt-en-ss",
    "Helsinki-NLP/opus-mt-en-st",
    "Helsinki-NLP/opus-mt-sv-en",
    "Helsinki-NLP/opus-mt-en-sw",
    "Helsinki-NLP/opus-mt-en-swc",
    "Helsinki-NLP/opus-mt-en-tdt",
    "Helsinki-NLP/opus-mt-en-ti",
    "Helsinki-NLP/opus-mt-en-tiv",
    "Helsinki-NLP/opus-mt-en-tl",
    "Helsinki-NLP/opus-mt-en-tll",
    "Helsinki-NLP/opus-mt-en-tn",
    "Helsinki-NLP/opus-mt-en-to",
    "Helsinki-NLP/opus-mt-en-toi",
    "Helsinki-NLP/opus-mt-en-tpi",
    "Helsinki-NLP/opus-mt-en-ts",
    "Helsinki-NLP/opus-mt-en-tut",
    "Helsinki-NLP/opus-mt-en-tvl",
    "Helsinki-NLP/opus-mt-en-tw",
    "Helsinki-NLP/opus-mt-en-ty",
    "Helsinki-NLP/opus-mt-en-umb",
    "Helsinki-NLP/opus-mt-en-ur",
    "Helsinki-NLP/opus-mt-en-urj",
    "Helsinki-NLP/opus-mt-en-xh",
    "Helsinki-NLP/opus-mt-en-zle",
    "Helsinki-NLP/opus-mt-en-zls",
    "Helsinki-NLP/opus-mt-en-zlw",
    "Helsinki-NLP/opus-mt-eo-es",
    "Helsinki-NLP/opus-mt-eo-fi",
    "Helsinki-NLP/opus-mt-eo-fr",
    "Helsinki-NLP/opus-mt-eo-he",
    "Helsinki-NLP/opus-mt-eo-hu",
    "Helsinki-NLP/opus-mt-eo-it",
    "Helsinki-NLP/opus-mt-eo-nl",
    "Helsinki-NLP/opus-mt-eo-pl",
    "Helsinki-NLP/opus-mt-eo-pt",
    "Helsinki-NLP/opus-mt-eo-ro",
    "Helsinki-NLP/opus-mt-eo-ru",
    "Helsinki-NLP/opus-mt-eo-sh",
    "Helsinki-NLP/opus-mt-eo-sv",
    "Helsinki-NLP/opus-mt-es-NORWAY",
    "Helsinki-NLP/opus-mt-es-ar",
    "Helsinki-NLP/opus-mt-es-ca",
    "Helsinki-NLP/opus-mt-es-cs",
    "Helsinki-NLP/opus-mt-es-efi",
    "Helsinki-NLP/opus-mt-es-el",
    "Helsinki-NLP/opus-mt-es-es",
    "Helsinki-NLP/opus-mt-es-et",
    "Helsinki-NLP/opus-mt-es-eu",
    "Helsinki-NLP/opus-mt-es-fi",
    "Helsinki-NLP/opus-mt-es-fj",
    "Helsinki-NLP/opus-mt-es-fr",
    "Helsinki-NLP/opus-mt-es-gaa",
    "Helsinki-NLP/opus-mt-es-gil",
    "Helsinki-NLP/opus-mt-es-gl",
    "Helsinki-NLP/opus-mt-es-guw",
    "Helsinki-NLP/opus-mt-es-ha",
    "Helsinki-NLP/opus-mt-es-he",
    "Helsinki-NLP/opus-mt-es-hil",
    "Helsinki-NLP/opus-mt-es-ho",
    "Helsinki-NLP/opus-mt-es-hr",
    "Helsinki-NLP/opus-mt-es-ht",
    "Helsinki-NLP/opus-mt-es-id",
    "Helsinki-NLP/opus-mt-es-ig",
    "Helsinki-NLP/opus-mt-es-ilo",
    "Helsinki-NLP/opus-mt-es-is",
    "Helsinki-NLP/opus-mt-es-iso",
    "Helsinki-NLP/opus-mt-es-it",
    "Helsinki-NLP/opus-mt-es-kg",
    "Helsinki-NLP/opus-mt-es-ln",
    "Helsinki-NLP/opus-mt-es-loz",
    "Helsinki-NLP/opus-mt-lt-es",
    "Helsinki-NLP/opus-mt-es-lua",
    "Helsinki-NLP/opus-mt-es-lus",
    "Helsinki-NLP/opus-mt-es-mfs",
    "Helsinki-NLP/opus-mt-es-mk",
    "Helsinki-NLP/opus-mt-es-mt",
    "Helsinki-NLP/opus-mt-es-niu",
    "Helsinki-NLP/opus-mt-es-nl",
    "Helsinki-NLP/opus-mt-es-no",
    "Helsinki-NLP/opus-mt-es-nso",
    "Helsinki-NLP/opus-mt-es-ny",
    "Helsinki-NLP/opus-mt-es-pag",
    "Helsinki-NLP/opus-mt-es-pap",
    "Helsinki-NLP/opus-mt-es-pis",
    "Helsinki-NLP/opus-mt-es-pl",
    "Helsinki-NLP/opus-mt-es-pon",
    "Helsinki-NLP/opus-mt-es-prl",
    "Helsinki-NLP/opus-mt-es-rn",
    "Helsinki-NLP/opus-mt-es-ro",
    "Helsinki-NLP/opus-mt-ru-es",
    "Helsinki-NLP/opus-mt-es-rw",
    "Helsinki-NLP/opus-mt-es-sg",
    "Helsinki-NLP/opus-mt-es-sl",
    "Helsinki-NLP/opus-mt-es-sm",
    "Helsinki-NLP/opus-mt-es-sn",
    "Helsinki-NLP/opus-mt-es-srn",
    "Helsinki-NLP/opus-mt-es-st",
    "Helsinki-NLP/opus-mt-es-swc",
    "Helsinki-NLP/opus-mt-es-tl",
    "Helsinki-NLP/opus-mt-es-tll",
    "Helsinki-NLP/opus-mt-es-tn",
    "Helsinki-NLP/opus-mt-es-to",
    "Helsinki-NLP/opus-mt-es-tpi",
    "Helsinki-NLP/opus-mt-es-tvl",
    "Helsinki-NLP/opus-mt-es-tw",
    "Helsinki-NLP/opus-mt-es-ty",
    "Helsinki-NLP/opus-mt-es-tzo",
    "Helsinki-NLP/opus-mt-es-uk",
    "Helsinki-NLP/opus-mt-es-ve",
    "Helsinki-NLP/opus-mt-vi-es",
    "Helsinki-NLP/opus-mt-es-war",
    "Helsinki-NLP/opus-mt-es-wls",
    "Helsinki-NLP/opus-mt-es-xh",
    "Helsinki-NLP/opus-mt-es-yo",
    "Helsinki-NLP/opus-mt-es-yua",
    "Helsinki-NLP/opus-mt-es-zai",
    "Helsinki-NLP/opus-mt-et-fi",
    "Helsinki-NLP/opus-mt-et-fr",
    "Helsinki-NLP/opus-mt-et-ru",
    "Helsinki-NLP/opus-mt-et-sv",
    "Helsinki-NLP/opus-mt-eu-en",
    "Helsinki-NLP/opus-mt-eu-ru",
    "Helsinki-NLP/opus-mt-fi-NORWAY",
    "Helsinki-NLP/opus-mt-fi-ZH",
    "Helsinki-NLP/opus-mt-fi-de",
    "Helsinki-NLP/opus-mt-fi-en",
    "Helsinki-NLP/opus-mt-fi-fi",
    "Helsinki-NLP/opus-mt-fi-fj",
    "Helsinki-NLP/opus-mt-fi-fr",
    "Helsinki-NLP/opus-mt-fi-fse",
    "Helsinki-NLP/opus-mt-fi-gaa",
    "Helsinki-NLP/opus-mt-fi-gil",
    "Helsinki-NLP/opus-mt-fi-guw",
    "Helsinki-NLP/opus-mt-fi-ha",
    "Helsinki-NLP/opus-mt-fi-he",
    "Helsinki-NLP/opus-mt-fi-ho",
    "Helsinki-NLP/opus-mt-fi-hr",
    "Helsinki-NLP/opus-mt-fi-ht",
    "Helsinki-NLP/opus-mt-fi-hu",
    "Helsinki-NLP/opus-mt-fi-id",
    "Helsinki-NLP/opus-mt-fi-ig",
    "Helsinki-NLP/opus-mt-fi-ilo",
    "Helsinki-NLP/opus-mt-fi-is",
    "Helsinki-NLP/opus-mt-fi-iso",
    "Helsinki-NLP/opus-mt-fi-it",
    "Helsinki-NLP/opus-mt-fi-kg",
    "Helsinki-NLP/opus-mt-fi-kqn",
    "Helsinki-NLP/opus-mt-fi-lg",
    "Helsinki-NLP/opus-mt-fi-ln",
    "Helsinki-NLP/opus-mt-fi-lu",
    "Helsinki-NLP/opus-mt-fi-lua",
    "Helsinki-NLP/opus-mt-fi-lue",
    "Helsinki-NLP/opus-mt-fi-lus",
    "Helsinki-NLP/opus-mt-fi-lv",
    "Helsinki-NLP/opus-mt-fi-mfe",
    "Helsinki-NLP/opus-mt-fi-mg",
    "Helsinki-NLP/opus-mt-fi-mh",
    "Helsinki-NLP/opus-mt-fi-mk",
    "Helsinki-NLP/opus-mt-fi-mos",
    "Helsinki-NLP/opus-mt-fi-mt",
    "Helsinki-NLP/opus-mt-fi-niu",
    "Helsinki-NLP/opus-mt-fi-nl",
    "Helsinki-NLP/opus-mt-fi-no",
    "Helsinki-NLP/opus-mt-fi-nso",
    "Helsinki-NLP/opus-mt-fi-ny",
    "Helsinki-NLP/opus-mt-fi-pag",
    "Helsinki-NLP/opus-mt-fi-pap",
    "Helsinki-NLP/opus-mt-fi-pis",
    "Helsinki-NLP/opus-mt-fi-pon",
    "Helsinki-NLP/opus-mt-fi-ro",
    "Helsinki-NLP/opus-mt-fi-ru",
    "Helsinki-NLP/opus-mt-fi-rw",
    "Helsinki-NLP/opus-mt-fi-sg",
    "Helsinki-NLP/opus-mt-fi-sk",
    "Helsinki-NLP/opus-mt-fi-sl",
    "Helsinki-NLP/opus-mt-fi-sm",
    "Helsinki-NLP/opus-mt-fi-sn",
    "Helsinki-NLP/opus-mt-fi-sq",
    "Helsinki-NLP/opus-mt-fi-srn",
    "Helsinki-NLP/opus-mt-fi-st",
    "Helsinki-NLP/opus-mt-fi-sv",
    "Helsinki-NLP/opus-mt-fi-sw",
    "Helsinki-NLP/opus-mt-fi-swc",
    "Helsinki-NLP/opus-mt-fi-tiv",
    "Helsinki-NLP/opus-mt-fi-tll",
    "Helsinki-NLP/opus-mt-fi-tn",
    "Helsinki-NLP/opus-mt-fi-to",
    "Helsinki-NLP/opus-mt-fi-toi",
    "Helsinki-NLP/opus-mt-fi-tpi",
    "Helsinki-NLP/opus-mt-fi-tr",
    "Helsinki-NLP/opus-mt-fi-ts",
    "Helsinki-NLP/opus-mt-fi-tvl",
    "Helsinki-NLP/opus-mt-fi-tw",
    "Helsinki-NLP/opus-mt-fi-ty",
    "Helsinki-NLP/opus-mt-fi-uk",
    "Helsinki-NLP/opus-mt-fi-ve",
    "Helsinki-NLP/opus-mt-fi-war",
    "Helsinki-NLP/opus-mt-fi-wls",
    "Helsinki-NLP/opus-mt-fi-xh",
    "Helsinki-NLP/opus-mt-fi-yap",
    "Helsinki-NLP/opus-mt-fi-yo",
    "Helsinki-NLP/opus-mt-fi-zne",
    "Helsinki-NLP/opus-mt-fiu-en",
    "Helsinki-NLP/opus-mt-fiu-fiu",
    "Helsinki-NLP/opus-mt-fj-fr",
    "Helsinki-NLP/opus-mt-fr-ar",
    "Helsinki-NLP/opus-mt-fr-gaa",
    "Helsinki-NLP/opus-mt-fr-gil",
    "Helsinki-NLP/opus-mt-fr-guw",
    "Helsinki-NLP/opus-mt-fr-ha",
    "Helsinki-NLP/opus-mt-fr-he",
    "Helsinki-NLP/opus-mt-fr-hil",
    "Helsinki-NLP/opus-mt-fr-ho",
    "Helsinki-NLP/opus-mt-fr-hr",
    "Helsinki-NLP/opus-mt-fr-ht",
    "Helsinki-NLP/opus-mt-fr-hu",
    "Helsinki-NLP/opus-mt-fr-id",
    "Helsinki-NLP/opus-mt-fr-ig",
    "Helsinki-NLP/opus-mt-fr-ilo",
    "Helsinki-NLP/opus-mt-fr-iso",
    "Helsinki-NLP/opus-mt-fr-kg",
    "Helsinki-NLP/opus-mt-fr-kqn",
    "Helsinki-NLP/opus-mt-fr-kwy",
    "Helsinki-NLP/opus-mt-fr-lg",
    "Helsinki-NLP/opus-mt-fr-ln",
    "Helsinki-NLP/opus-mt-fr-loz",
    "Helsinki-NLP/opus-mt-fr-lu",
    "Helsinki-NLP/opus-mt-fr-lua",
    "Helsinki-NLP/opus-mt-fr-lue",
    "Helsinki-NLP/opus-mt-fr-lus",
    "Helsinki-NLP/opus-mt-fr-mfe",
    "Helsinki-NLP/opus-mt-fr-mh",
    "Helsinki-NLP/opus-mt-fr-mos",
    "Helsinki-NLP/opus-mt-fr-ms",
    "Helsinki-NLP/opus-mt-fr-mt",
    "Helsinki-NLP/opus-mt-fr-niu",
    "Helsinki-NLP/opus-mt-fr-no",
    "Helsinki-NLP/opus-mt-fr-nso",
    "Helsinki-NLP/opus-mt-fr-ny",
    "Helsinki-NLP/opus-mt-fr-pag",
    "Helsinki-NLP/opus-mt-fr-pap",
    "Helsinki-NLP/opus-mt-fr-pis",
    "Helsinki-NLP/opus-mt-pl-fr",
    "Helsinki-NLP/opus-mt-fr-pon",
    "Helsinki-NLP/opus-mt-fr-rnd",
    "Helsinki-NLP/opus-mt-fr-ro",
    "Helsinki-NLP/opus-mt-fr-ru",
    "Helsinki-NLP/opus-mt-fr-run",
    "Helsinki-NLP/opus-mt-fr-rw",
    "Helsinki-NLP/opus-mt-fr-sg",
    "Helsinki-NLP/opus-mt-fr-sk",
    "Helsinki-NLP/opus-mt-fr-sl",
    "Helsinki-NLP/opus-mt-fr-sm",
    "Helsinki-NLP/opus-mt-fr-sn",
    "Helsinki-NLP/opus-mt-fr-srn",
    "Helsinki-NLP/opus-mt-fr-st",
    "Helsinki-NLP/opus-mt-fr-sv",
    "Helsinki-NLP/opus-mt-fr-swc",
    "Helsinki-NLP/opus-mt-fr-tiv",
    "Helsinki-NLP/opus-mt-fr-tl",
    "Helsinki-NLP/opus-mt-fr-tll",
    "Helsinki-NLP/opus-mt-fr-tn",
    "Helsinki-NLP/opus-mt-fr-to",
    "Helsinki-NLP/opus-mt-fr-tpi",
    "Helsinki-NLP/opus-mt-fr-ts",
    "Helsinki-NLP/opus-mt-fr-tum",
    "Helsinki-NLP/opus-mt-fr-tvl",
    "Helsinki-NLP/opus-mt-fr-tw",
    "Helsinki-NLP/opus-mt-fr-ty",
    "Helsinki-NLP/opus-mt-fr-uk",
    "Helsinki-NLP/opus-mt-fr-ve",
    "Helsinki-NLP/opus-mt-fr-vi",
    "Helsinki-NLP/opus-mt-fr-war",
    "Helsinki-NLP/opus-mt-fr-wls",
    "Helsinki-NLP/opus-mt-fr-xh",
    "Helsinki-NLP/opus-mt-fr-yap",
    "Helsinki-NLP/opus-mt-fr-yo",
    "Helsinki-NLP/opus-mt-fr-zne",
    "Helsinki-NLP/opus-mt-gaa-sv",
    "Helsinki-NLP/opus-mt-gem-en",
    "Helsinki-NLP/opus-mt-gem-gem",
    "Helsinki-NLP/opus-mt-gil-sv",
    "Helsinki-NLP/opus-mt-gl-en",
    "Helsinki-NLP/opus-mt-gl-pt",
    "Helsinki-NLP/opus-mt-gmq-en",
    "Helsinki-NLP/opus-mt-gmq-gmq",
    "Helsinki-NLP/opus-mt-gmw-en",
    "Helsinki-NLP/opus-mt-gmw-gmw",
    "Helsinki-NLP/opus-mt-grk-en",
    "Helsinki-NLP/opus-mt-guw-sv",
    "Helsinki-NLP/opus-mt-ha-sv",
    "Helsinki-NLP/opus-mt-he-ar",
    "Helsinki-NLP/opus-mt-he-it",
    "Helsinki-NLP/opus-mt-he-ru",
    "Helsinki-NLP/opus-mt-he-sv",
    "Helsinki-NLP/opus-mt-he-uk",
    "Helsinki-NLP/opus-mt-hi-en",
    "Helsinki-NLP/opus-mt-hi-ur",
    "Helsinki-NLP/opus-mt-hil-fi",
    "Helsinki-NLP/opus-mt-hr-sv",
    "Helsinki-NLP/opus-mt-ht-sv",
    "Helsinki-NLP/opus-mt-hu-sv",
    "Helsinki-NLP/opus-mt-hu-uk",
    "Helsinki-NLP/opus-mt-hy-en",
    "Helsinki-NLP/opus-mt-hy-ru",
    "Helsinki-NLP/opus-mt-id-sv",
    "Helsinki-NLP/opus-mt-ig-sv",
    "Helsinki-NLP/opus-mt-iir-en",
    "Helsinki-NLP/opus-mt-iir-iir",
    "Helsinki-NLP/opus-mt-ilo-en",
    "Helsinki-NLP/opus-mt-ilo-sv",
    "Helsinki-NLP/opus-mt-inc-en",
    "Helsinki-NLP/opus-mt-inc-inc",
    "Helsinki-NLP/opus-mt-ine-en",
    "Helsinki-NLP/opus-mt-ine-ine",
    "Helsinki-NLP/opus-mt-is-eo",
    "Helsinki-NLP/opus-mt-is-fr",
    "Helsinki-NLP/opus-mt-is-it",
    "Helsinki-NLP/opus-mt-is-sv",
    "Helsinki-NLP/opus-mt-iso-sv",
    "Helsinki-NLP/opus-mt-it-en",
    "Helsinki-NLP/opus-mt-it-fr",
    "Helsinki-NLP/opus-mt-it-lt",
    "Helsinki-NLP/opus-mt-it-ms",
    "Helsinki-NLP/opus-mt-it-sv",
    "Helsinki-NLP/opus-mt-it-uk",
    "Helsinki-NLP/opus-mt-it-vi",
    "Helsinki-NLP/opus-mt-itc-en",
    "Helsinki-NLP/opus-mt-itc-itc",
    "Helsinki-NLP/opus-mt-ja-ar",
    "Helsinki-NLP/opus-mt-ja-bg",
    "Helsinki-NLP/opus-mt-ja-da",
    "Helsinki-NLP/opus-mt-ja-de",
    "Helsinki-NLP/opus-mt-ja-fi",
    "Helsinki-NLP/opus-mt-ja-fr",
    "Helsinki-NLP/opus-mt-ja-he",
    "Helsinki-NLP/opus-mt-ja-hu",
    "Helsinki-NLP/opus-mt-ja-it",
    "Helsinki-NLP/opus-mt-ja-ms",
    "Helsinki-NLP/opus-mt-ja-nl",
    "Helsinki-NLP/opus-mt-ja-pl",
    "Helsinki-NLP/opus-mt-ja-pt",
    "Helsinki-NLP/opus-mt-ja-ru",
    "Helsinki-NLP/opus-mt-ja-sh",
    "Helsinki-NLP/opus-mt-ja-sv",
    "Helsinki-NLP/opus-mt-ja-tr",
    "Helsinki-NLP/opus-mt-ja-vi",
    "Helsinki-NLP/opus-mt-ka-en",
    "Helsinki-NLP/opus-mt-ka-ru",
    "Helsinki-NLP/opus-mt-kab-en",
    "Helsinki-NLP/opus-mt-kg-sv",
    "Helsinki-NLP/opus-mt-kl-en",
    "Helsinki-NLP/opus-mt-ko-de",
    "Helsinki-NLP/opus-mt-ko-es",
    "Helsinki-NLP/opus-mt-ko-fi",
    "Helsinki-NLP/opus-mt-ko-fr",
    "Helsinki-NLP/opus-mt-ko-hu",
    "Helsinki-NLP/opus-mt-ko-ru",
    "Helsinki-NLP/opus-mt-ko-sv",
    "Helsinki-NLP/opus-mt-kqn-es",
    "Helsinki-NLP/opus-mt-kqn-sv",
    "Helsinki-NLP/opus-mt-kwy-sv",
    "Helsinki-NLP/opus-mt-lg-es",
    "Helsinki-NLP/opus-mt-lg-sv",
    "Helsinki-NLP/opus-mt-loz-fi",
    "Helsinki-NLP/opus-mt-loz-sv",
    "Helsinki-NLP/opus-mt-lt-eo",
    "Helsinki-NLP/opus-mt-lt-fr",
    "Helsinki-NLP/opus-mt-lt-pl",
    "Helsinki-NLP/opus-mt-lt-ru",
    "Helsinki-NLP/opus-mt-lt-sv",
    "Helsinki-NLP/opus-mt-lt-tr",
    "Helsinki-NLP/opus-mt-lu-es",
    "Helsinki-NLP/opus-mt-lu-sv",
    "Helsinki-NLP/opus-mt-lua-sv",
    "Helsinki-NLP/opus-mt-lue-es",
    "Helsinki-NLP/opus-mt-lue-sv",
    "Helsinki-NLP/opus-mt-lus-sv",
    "Helsinki-NLP/opus-mt-lv-en",
    "Helsinki-NLP/opus-mt-lv-es",
    "Helsinki-NLP/opus-mt-lv-fr",
    "Helsinki-NLP/opus-mt-lv-ru",
    "Helsinki-NLP/opus-mt-lv-sv",
    "Helsinki-NLP/opus-mt-mfe-es",
    "Helsinki-NLP/opus-mt-mg-es",
    "Helsinki-NLP/opus-mt-mh-es",
    "Helsinki-NLP/opus-mt-mk-fr",
    "Helsinki-NLP/opus-mt-mkh-en",
    "Helsinki-NLP/opus-mt-ml-en",
    "Helsinki-NLP/opus-mt-ms-de",
    "Helsinki-NLP/opus-mt-ms-fr",
    "Helsinki-NLP/opus-mt-ms-it",
    "Helsinki-NLP/opus-mt-ms-ms",
    "Helsinki-NLP/opus-mt-mt-sv",
    "Helsinki-NLP/opus-mt-nic-en",
    "Helsinki-NLP/opus-mt-niu-sv",
    "Helsinki-NLP/opus-mt-nl-fr",
    "Helsinki-NLP/opus-mt-nl-no",
    "Helsinki-NLP/opus-mt-nl-sv",
    "Helsinki-NLP/opus-mt-nl-uk",
    "Helsinki-NLP/opus-mt-no-da",
    "Helsinki-NLP/opus-mt-no-de",
    "Helsinki-NLP/opus-mt-no-es",
    "Helsinki-NLP/opus-mt-no-fi",
    "Helsinki-NLP/opus-mt-no-fr",
    "Helsinki-NLP/opus-mt-no-no",
    "Helsinki-NLP/opus-mt-no-pl",
    "Helsinki-NLP/opus-mt-no-ru",
    "Helsinki-NLP/opus-mt-no-sv",
    "Helsinki-NLP/opus-mt-no-uk",
    "Helsinki-NLP/opus-mt-nso-sv",
    "Helsinki-NLP/opus-mt-pa-en",
    "Helsinki-NLP/opus-mt-pag-sv",
    "Helsinki-NLP/opus-mt-phi-en",
    "Helsinki-NLP/opus-mt-pis-sv",
    "Helsinki-NLP/opus-mt-pl-ar",
    "Helsinki-NLP/opus-mt-pl-en",
    "Helsinki-NLP/opus-mt-pl-sv",
    "Helsinki-NLP/opus-mt-pl-uk",
    "Helsinki-NLP/opus-mt-pon-sv",
    "Helsinki-NLP/opus-mt-pqe-en",
    "Helsinki-NLP/opus-mt-pt-tl",
    "Helsinki-NLP/opus-mt-pt-uk",
    "Helsinki-NLP/opus-mt-rn-de",
    "Helsinki-NLP/opus-mt-rn-fr",
    "Helsinki-NLP/opus-mt-rn-ru",
    "Helsinki-NLP/opus-mt-rnd-sv",
    "Helsinki-NLP/opus-mt-ro-sv",
    "Helsinki-NLP/opus-mt-ru-ar",
    "Helsinki-NLP/opus-mt-ru-bg",
    "Helsinki-NLP/opus-mt-ru-hy",
    "Helsinki-NLP/opus-mt-ru-no",
    "Helsinki-NLP/opus-mt-ru-sl",
    "Helsinki-NLP/opus-mt-ru-sv",
    "Helsinki-NLP/opus-mt-ru-uk",
    "Helsinki-NLP/opus-mt-ru-vi",
    "Helsinki-NLP/opus-mt-run-es",
    "Helsinki-NLP/opus-mt-run-sv",
    "Helsinki-NLP/opus-mt-rw-sv",
    "Helsinki-NLP/opus-mt-sal-en",
    "Helsinki-NLP/opus-mt-sem-en",
    "Helsinki-NLP/opus-mt-sem-sem",
    "Helsinki-NLP/opus-mt-sg-sv",
    "Helsinki-NLP/opus-mt-sh-eo",
    "Helsinki-NLP/opus-mt-sh-uk",
    "Helsinki-NLP/opus-mt-sk-es",
    "Helsinki-NLP/opus-mt-sk-sv",
    "Helsinki-NLP/opus-mt-sl-sv",
    "Helsinki-NLP/opus-mt-sl-uk",
    "Helsinki-NLP/opus-mt-sla-en",
    "Helsinki-NLP/opus-mt-sla-sla",
    "Helsinki-NLP/opus-mt-sn-sv",
    "Helsinki-NLP/opus-mt-sq-es",
    "Helsinki-NLP/opus-mt-sq-sv",
    "Helsinki-NLP/opus-mt-srn-en",
    "Helsinki-NLP/opus-mt-srn-sv",
    "Helsinki-NLP/opus-mt-ssp-es",
    "Helsinki-NLP/opus-mt-st-sv",
    "Helsinki-NLP/opus-mt-sv-NORWAY",
    "Helsinki-NLP/opus-mt-sv-ZH",
    "Helsinki-NLP/opus-mt-sv-es",
    "Helsinki-NLP/opus-mt-sv-fi",
    "Helsinki-NLP/opus-mt-sv-fj",
    "Helsinki-NLP/opus-mt-sv-hil",
    "Helsinki-NLP/opus-mt-sv-ho",
    "Helsinki-NLP/opus-mt-sv-ln",
    "Helsinki-NLP/opus-mt-sv-mfe",
    "Helsinki-NLP/opus-mt-sv-mh",
    "Helsinki-NLP/opus-mt-sv-mos",
    "Helsinki-NLP/opus-mt-sv-no",
    "Helsinki-NLP/opus-mt-sv-ny",
    "Helsinki-NLP/opus-mt-sv-pap",
    "Helsinki-NLP/opus-mt-sv-ru",
    "Helsinki-NLP/opus-mt-sv-sm",
    "Helsinki-NLP/opus-mt-sv-sv",
    "Helsinki-NLP/opus-mt-sv-swc",
    "Helsinki-NLP/opus-mt-sv-th",
    "Helsinki-NLP/opus-mt-sv-tiv",
    "Helsinki-NLP/opus-mt-sv-tll",
    "Helsinki-NLP/opus-mt-sv-tn",
    "Helsinki-NLP/opus-mt-sv-to",
    "Helsinki-NLP/opus-mt-sv-toi",
    "Helsinki-NLP/opus-mt-sv-tpi",
    "Helsinki-NLP/opus-mt-sv-ts",
    "Helsinki-NLP/opus-mt-sv-tum",
    "Helsinki-NLP/opus-mt-sv-tvl",
    "Helsinki-NLP/opus-mt-sv-tw",
    "Helsinki-NLP/opus-mt-sv-ty",
    "Helsinki-NLP/opus-mt-sv-uk",
    "Helsinki-NLP/opus-mt-sv-umb",
    "Helsinki-NLP/opus-mt-sv-ve",
    "Helsinki-NLP/opus-mt-sv-war",
    "Helsinki-NLP/opus-mt-sv-wls",
    "Helsinki-NLP/opus-mt-sv-xh",
    "Helsinki-NLP/opus-mt-sv-yap",
    "Helsinki-NLP/opus-mt-sv-yo",
    "Helsinki-NLP/opus-mt-sv-zne",
    "Helsinki-NLP/opus-mt-taw-en",
    "Helsinki-NLP/opus-mt-tc-base-gmw-gmw",
    "Helsinki-NLP/opus-mt-th-en",
    "Helsinki-NLP/opus-mt-th-fr",
    "Helsinki-NLP/opus-mt-tl-en",
    "Helsinki-NLP/opus-mt-toi-es",
    "Helsinki-NLP/opus-mt-toi-fr",
    "Helsinki-NLP/opus-mt-tr-ar",
    "Helsinki-NLP/opus-mt-tr-eo",
    "Helsinki-NLP/opus-mt-tr-es",
    "Helsinki-NLP/opus-mt-tr-fr",
    "Helsinki-NLP/opus-mt-tr-sv",
    "Helsinki-NLP/opus-mt-tr-uk",
    "Helsinki-NLP/opus-mt-trk-en",
    "Helsinki-NLP/opus-mt-ts-es",
    "Helsinki-NLP/opus-mt-tum-en",
    "Helsinki-NLP/opus-mt-tum-es",
    "Helsinki-NLP/opus-mt-uk-es",
    "Helsinki-NLP/opus-mt-uk-sh",
    "Helsinki-NLP/opus-mt-urj-en",
    "Helsinki-NLP/opus-mt-urj-urj",
    "Helsinki-NLP/opus-mt-ve-en",
    "Helsinki-NLP/opus-mt-vi-eo",
    "Helsinki-NLP/opus-mt-vsl-es",
    "Helsinki-NLP/opus-mt-wal-en",
    "Helsinki-NLP/opus-mt-war-en",
    "Helsinki-NLP/opus-mt-wls-en",
    "Helsinki-NLP/opus-mt-yap-en",
    "Helsinki-NLP/opus-mt-yo-en",
    "Helsinki-NLP/opus-mt-zh-bg",
    "Helsinki-NLP/opus-mt-zh-de",
    "Helsinki-NLP/opus-mt-zh-fi",
    "Helsinki-NLP/opus-mt-zh-he",
    "Helsinki-NLP/opus-mt-zh-it",
    "Helsinki-NLP/opus-mt-zh-ms",
    "Helsinki-NLP/opus-mt-zh-nl",
    "Helsinki-NLP/opus-mt-zh-sv",
    "Helsinki-NLP/opus-mt-zh-uk",
    "Helsinki-NLP/opus-mt-zh-vi",
    "Helsinki-NLP/opus-mt-zle-en",
    "Helsinki-NLP/opus-mt-zle-zle",
    "Helsinki-NLP/opus-mt-zls-en",
    "Helsinki-NLP/opus-mt-zls-zls",
    "Helsinki-NLP/opus-mt-zlw-en",
    "Helsinki-NLP/opus-mt-zlw-fiu",
    "Helsinki-NLP/opus-mt-zlw-zlw",
    "Helsinki-NLP/opus-mt-zne-es",
    "Helsinki-NLP/opus-tatoeba-de-ro",
    "Helsinki-NLP/opus-tatoeba-en-ja",
    "Helsinki-NLP/opus-tatoeba-en-ro",
    "Helsinki-NLP/opus-tatoeba-en-tr",
    "Helsinki-NLP/opus-tatoeba-es-zh",
    "Helsinki-NLP/opus-tatoeba-fi-en",
    "Helsinki-NLP/opus-tatoeba-fr-it",
    "Helsinki-NLP/opus-mt-tc-big-zle-en",
    "Helsinki-NLP/opus-mt-tc-big-fi-en",
    "Helsinki-NLP/opus-mt-tc-big-en-fi",
    "Helsinki-NLP/opus-mt-tc-base-uk-ces_slk",
    "Helsinki-NLP/opus-mt-tc-base-uk-hu",
    "Helsinki-NLP/opus-mt-tc-base-uk-fi",
    "Helsinki-NLP/opus-mt-tc-base-uk-ro",
    "Helsinki-NLP/opus-mt-tc-base-uk-tr",
    "Helsinki-NLP/opus-mt-tc-big-zle-fi",
    "Helsinki-NLP/opus-mt-tc-base-zle-bat",
    "Helsinki-NLP/opus-mt-tc-big-zle-de",
    "Helsinki-NLP/opus-mt-tc-big-fi-zle",
    "Helsinki-NLP/opus-mt-tc-big-zle-fr",
    "Helsinki-NLP/opus-mt-tc-big-zle-gmq",
    "Helsinki-NLP/opus-mt-tc-big-zle-it",
    "Helsinki-NLP/opus-mt-tc-big-zle-pt",
    "Helsinki-NLP/opus-mt-tc-big-zle-es",
    "Helsinki-NLP/opus-mt-tc-big-zle-zle",
    "Helsinki-NLP/opus-mt-tc-big-zle-zls",
    "Helsinki-NLP/opus-mt-tc-big-zle-zlw",
    "Helsinki-NLP/opus-mt-tc-base-bat-zle",
    "Helsinki-NLP/opus-mt-tc-base-ces_slk-uk",
    "Helsinki-NLP/opus-mt-tc-big-de-zle",
    "Helsinki-NLP/opus-mt-tc-big-en-zle",
    "Helsinki-NLP/opus-mt-tc-base-fi-uk",
    "Helsinki-NLP/opus-mt-tc-big-fr-zle",
    "Helsinki-NLP/opus-mt-tc-big-gmq-zle",
    "Helsinki-NLP/opus-mt-tc-big-it-zle",
    "Helsinki-NLP/opus-mt-tc-big-pt-zle",
    "Helsinki-NLP/opus-mt-tc-base-ro-uk",
    "Helsinki-NLP/opus-mt-tc-big-es-zle",
    "Helsinki-NLP/opus-mt-tc-base-tr-uk",
    "Helsinki-NLP/opus-mt-tc-big-zls-zle",
    "Helsinki-NLP/opus-mt-tc-big-zlw-zle",
    "Helsinki-NLP/opus-mt-tc-big-en-bg",
    "Helsinki-NLP/opus-mt-tc-big-en-cat_oci_spa",
    "Helsinki-NLP/opus-mt-tc-big-en-ces_slk",
    "Helsinki-NLP/opus-mt-tc-big-el-en",
    "Helsinki-NLP/opus-mt-tc-big-en-et",
    "Helsinki-NLP/opus-mt-tc-big-en-fr",
    "Helsinki-NLP/opus-mt-tc-big-en-gmq",
    "Helsinki-NLP/opus-mt-tc-big-en-hu",
    "Helsinki-NLP/opus-mt-tc-big-en-it",
    "Helsinki-NLP/opus-mt-tc-big-en-lv",
    "Helsinki-NLP/opus-mt-tc-big-en-lt",
    "Helsinki-NLP/opus-mt-tc-big-en-ro",
    "Helsinki-NLP/opus-mt-tc-big-ar-en",
    "Helsinki-NLP/opus-mt-tc-big-bg-en",
    "Helsinki-NLP/opus-mt-tc-big-cat_oci_spa-en",
    "Helsinki-NLP/opus-mt-tc-big-cel-en",
    "Helsinki-NLP/opus-mt-tc-big-ces_slk-en",
    "Helsinki-NLP/opus-mt-tc-big-et-en",
    "Helsinki-NLP/opus-mt-tc-big-gmq-en",
    "Helsinki-NLP/opus-mt-tc-big-sh-en",
    "Helsinki-NLP/opus-mt-tc-big-lv-en",
    "Helsinki-NLP/opus-mt-tc-big-lt-en",
    "Helsinki-NLP/opus-mt-tc-big-zls-en",
    "Helsinki-NLP/opus-mt-tc-big-zlw-en",
    "Helsinki-NLP/opus-mt-tc-big-itc-itc",
    "Helsinki-NLP/opus-mt-tc-big-gmw-gmw",
    "Helsinki-NLP/opus-mt-tc-big-gmq-gmq",
    "Helsinki-NLP/opus-mt-tc-big-zls-itc",
    "Helsinki-NLP/opus-mt-tc-big-gmq-itc",
    "Helsinki-NLP/opus-mt-tc-big-itc-ar",
    "Helsinki-NLP/opus-mt-tc-big-ar-gmq",
    "Helsinki-NLP/opus-mt-tc-big-zle-itc",
    "Helsinki-NLP/opus-mt-tc-big-itc-he",
    "Helsinki-NLP/opus-mt-tc-big-gmq-zlw",
    "Helsinki-NLP/opus-mt-tc-big-zh-ja",
    "Helsinki-NLP/opus-mt-tc-big-itc-tr",
    "Helsinki-NLP/opus-mt-tc-big-gmq-he",
    "Helsinki-NLP/opus-mt-tc-big-ar-itc",
    "Helsinki-NLP/opus-mt-tc-big-itc-bat",
    "Helsinki-NLP/opus-mt-tc-big-gmq-ar",
    "Helsinki-NLP/opus-mt-tc-big-zls-de",
    "Helsinki-NLP/opus-mt-tc-big-gmq-tr",
    "Helsinki-NLP/opus-mt-tc-big-de-es",
    "Helsinki-NLP/opus-mt-tc-big-he-itc",
    "Helsinki-NLP/opus-mt-tc-big-de-gmq",
    "Helsinki-NLP/opus-mt-tc-big-he-gmq",
    "Helsinki-NLP/opus-mt-tc-big-itc-eu",
    "Helsinki-NLP/opus-mt-tc-big-fi-zls",
    "Helsinki-NLP/opus-mt-tc-big-fa-itc",
    "Helsinki-NLP/opus-mt-tc-big-fa-gmq",
    "Helsinki-NLP/opus-mt-tc-big-eu-itc",
    "Helsinki-NLP/opus-mt-tc-base-en-sh",
    "Helsinki-NLP/simple-finnish-gpt3-xl",
    "Helsinki-NLP/opus-mt-tc-bible-big-afa-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-sla-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-ine-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-bnt-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-aav-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-afa-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-afa-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-afa-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-alv-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-bat-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-bat-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-bat-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-cel-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-aav",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-afa",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-bat",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-bnt",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-fiu",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gem",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmq",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-iir",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-inc",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-ine",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-itc",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-mkh",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-pqw",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-roa",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-sem",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-sla",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-trk",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-urj",
    "Helsinki-NLP/opus-mt-tc-bible-big-dra-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-dra-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-fiu-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-fiu-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-fiu-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-gem-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-gem-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-gmq-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-gmq-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-gmw-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-gmw-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-iir-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-iir-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-inc-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-inc-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-inc-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-ine-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-ira-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-itc-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-itc-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-itc-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-map-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-map-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-mkh-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-mkh-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-mul-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-phi-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-phi-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-poz-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-poz-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-poz-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-pqe-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-pqw-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-pqw-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-pqw-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-roa-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-roa-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-sem-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-sem-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-sem-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-sla-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-tai-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-trk-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-urj-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-urj-deu_eng_nld",
    "Helsinki-NLP/opus-mt-tc-bible-big-urj-fra_ita_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-zhx-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-zhx-en",
    "Helsinki-NLP/opus-mt-tc-bible-big-mul-deu_eng_fra_por_spa",
    "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-mul",
    "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul",
]

kochi_yasuo_models = [
    "KoichiYasuoka/roberta-base-thai-spm-upos", # 2
    "KoichiYasuoka/bert-base-vietnamese-ud-goeswith", # 2
    "KoichiYasuoka/bert-base-thai-upos", # 1
    "KoichiYasuoka/roberta-base-japanese-aozora-char", # 1
    "KoichiYasuoka/roberta-base-thai-syllable-upos", # 1
    "KoichiYasuoka/bert-base-german-upos", # 1
    "KoichiYasuoka/roberta-base-serbian", # 1
    "KoichiYasuoka/deberta-small-japanese-upos", # 1
    "KoichiYasuoka/deberta-base-thai-ud-head", # 1
    "KoichiYasuoka/Swallow-MS-7b-upos", # 1
    "KoichiYasuoka/gpt2-small-japanese-ud-causal", # 1
    "KoichiYasuoka/Xunzi-Qwen2-7B-upos", # 1
    "KoichiYasuoka/Llama-3-Swallow-8B-upos", # 1
    "KoichiYasuoka/RakutenAI-7B-upos", # 1
    "KoichiYasuoka/bert-base-tibetan", # 1
    "KoichiYasuoka/modernbert-large-japanese-aozora", # 1
    "KoichiYasuoka/modernbert-base-japanese-char", # 1
    "KoichiYasuoka/roberta-base-thai-char-upos", # 0
    "KoichiYasuoka/roberta-small-coptic", # 0
    "KoichiYasuoka/roberta-base-coptic", # 0
    "KoichiYasuoka/deberta-small-coptic", # 0
    "KoichiYasuoka/deberta-base-japanese-unidic", # 0
    "KoichiYasuoka/roberta-base-latin-ud-goeswith", # 0
    "KoichiYasuoka/roberta-base-vietnamese", # 0
    "KoichiYasuoka/phobert-base-vietnamese-ud-goeswith", # 0
    "KoichiYasuoka/deberta-base-chinese-ud-goeswith", # 0
    "KoichiYasuoka/deberta-base-ainu", # 0
    "KoichiYasuoka/roberta-base-ainu", # 0
    "KoichiYasuoka/deberta-base-japanese-juman-ud-goeswith", # 0
    "KoichiYasuoka/roberta-base-japanese-juman-ud-goeswith", # 0
    "KoichiYasuoka/deberta-base-korean-upos", # 0
    "KoichiYasuoka/Swallow-7b-plus-upos", # 0
    "KoichiYasuoka/deberta-v3-base-japanese-ud-goeswith", # 0
    "KoichiYasuoka/deberta-small-occitan", # 0
    "KoichiYasuoka/deberta-base-belarusian", # 0
    "KoichiYasuoka/Xunzi-Qwen2-1.5B-ud-causal", # 0
    "KoichiYasuoka/llm-jp-1.3b-upos", # 0
    "KoichiYasuoka/karasu-1.1B-upos", # 0
    "KoichiYasuoka/gpt2-medium-japanese-unidic-upos", # 0
    "KoichiYasuoka/gpt2-small-japanese-juman-upos", # 0
    "KoichiYasuoka/gpt2-xlarge-japanese-juman-upos", # 0
    "KoichiYasuoka/rinna-gpt2-xsmall-japanese-ud-causal", # 0
    "KoichiYasuoka/open-calm-small-ud-causal", # 0
    "KoichiYasuoka/rinna-gpt-neox-small-japanese-ud-causal", # 0
    "KoichiYasuoka/abeja-gpt2-large-japanese-ud-causal", # 0
    "KoichiYasuoka/goldfish-gpt2-japanese-5mb-ud-causal", # 0
    "KoichiYasuoka/goldfish-gpt2-japanese-10mb-ud-causal", # 0
    "KoichiYasuoka/goldfish-gpt2-japanese-100mb-ud-causal", # 0
    "KoichiYasuoka/goldfish-gpt2-thai-ud-causal", # 0
    "KoichiYasuoka/camembert-thai-base", # 0
    "KoichiYasuoka/ltgbert-base-japanese-upos", # 0
    "KoichiYasuoka/ltgbert-base-belarusian-upos", # 0
    "KoichiYasuoka/tibetan-bert-base-upos", # 0
    "KoichiYasuoka/roberta-base-tibetan", # 0
    "KoichiYasuoka/yellowback-gpt-neo-1.3B-ud-causal", # 0
    "KoichiYasuoka/gpt2-base-japanese-upos", # 0
    "KoichiYasuoka/rinna-gpt2-1b-japanese-ud-causal", # 0
    "KoichiYasuoka/stockmark-gpt-neox-1.4b-japanese-ud-causal", # 0
    "KoichiYasuoka/gpt2-small-serbian-upos", # 0
    "KoichiYasuoka/modernbert-base-english-ud-triangular", # 0
    "KoichiYasuoka/modernbert-base-thai-wikipedia", # 0
    "KoichiYasuoka/modernbert-small-japanese-aozora", # 0
    "KoichiYasuoka/modernbert-japanese-130m-ud-embeds", # 0
    "KoichiYasuoka/modernbert-base-japanese-unidic-ud-triangular", # 0
    "KoichiYasuoka/llm-jp-3-150m-ud-embeds", # 0
    "KoichiYasuoka/modernbert-base-turkish-ud-embeds", # 0
    "KoichiYasuoka/roberta-base-turkish-ud-goeswith", # 0
    "KoichiYasuoka/ltgbert-base-turkish-ud-goeswith", # 0
    "KoichiYasuoka/modernbert-base-classical-chinese-traditional", # 0
    "KoichiYasuoka/modernbert-base-classical-chinese", # 0
]

MODEL_NAME_TO_LIKES = {}
MODEL_NAME_TO_CREATED_AT = {}
MODEL_NAME_TO_TOKENIZER_HASHES = {}
TEMP_MODEL_NAME_TO_TOKENIZER_HASHES_FILE = 'data/temp_model_name_to_tokenizer_hashes.json'
TEMP_MODEL_LIKES_AND_CREATED_AT_FILE = 'data/temp_model_likes_and_created_at.json'
BLACKLIST_FILE = 'data/blacklist.txt'
BLACKLIST_MODEL_NAMES = set()
import datetime
import json
import os

def get_evaluated_tokenizer_hashes_and_likes(results_dir, verbose=False):
    """Extract tokenizer_hash and likes from already evaluated models."""
    from pathlib import Path
    import json
    from collections import defaultdict
    from tokenizer_evaluate import get_tokenizer_hash_field, generate_model_key
    
    tokenizer_hash_field = get_tokenizer_hash_field()
    results_path = Path(results_dir) / "evaluations"
    if not results_path.exists():
        return {}, set(), set()
    
    all_model_names = set()
    all_model_keys = set()
    tokenizer_hash_to_best_likes = {}  # tokenizer_hash -> (model_key, likes, created_at)
    
    for result_file in results_path.glob("results_*.jsonl"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    data = json.loads(first_line)
                    
                    # Extract model configuration
                    model_name = data.get('model')
                    revision = data.get('revision', 'main')
                    subfolder = data.get('subfolder', None)
                    _target_mode = data.get('_target_mode', False)
                    
                    # Generate model key for this configuration
                    model_key = generate_model_key(model_name, revision, subfolder, _target_mode)
                    
                    tokenizer_hash = data.get(tokenizer_hash_field)
                    likes = data.get('likes', 0)
                    created_at = data.get('created_at', MODEL_NAME_TO_CREATED_AT.get(model_name, datetime.datetime.now().isoformat()))
                    
                    # Track both model names and full model keys
                    if model_name:
                        all_model_names.add(model_name)
                    all_model_keys.add(model_key)
                    
                    if tokenizer_hash and model_name is not None:
                        # Keep the model with most likes for each tokenizer_hash
                        if tokenizer_hash not in tokenizer_hash_to_best_likes or likes > tokenizer_hash_to_best_likes[tokenizer_hash][1]:
                            tokenizer_hash_to_best_likes[tokenizer_hash] = (model_key, likes, created_at)
                            
        except Exception as e:
            if verbose:
                print(f"Warning: Could not read {result_file}: {e}")
            continue
    
    if verbose:
        print(f"Found {len(tokenizer_hash_to_best_likes)} unique tokenizer_hashes in evaluated models")
        print(f"Found {len(all_model_keys)} unique model_keys in evaluated models")
        
    return tokenizer_hash_to_best_likes, all_model_names, all_model_keys

# Helper function to normalize model inputs
def save_likes_and_created_at_cache():
    """Save MODEL_NAME_TO_LIKES and MODEL_NAME_TO_CREATED_AT to temporary file."""
    try:
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(TEMP_MODEL_LIKES_AND_CREATED_AT_FILE), exist_ok=True)
        
        cache_data = {
            'likes': MODEL_NAME_TO_LIKES,
            'created_at': MODEL_NAME_TO_CREATED_AT
        }
        with open(TEMP_MODEL_LIKES_AND_CREATED_AT_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Warning: Could not save likes and created_at cache: {e}")

def normalize_model_input(model_input):
    """
    Normalize model input to a standardized tuple format.
    
    Args:
        model_input: Either a string (model_name) or tuple (model_name, revision, subfolder, _target_mode)
        
    Returns:
        tuple: (model_name, revision, subfolder, _target_mode) with defaults applied
    """
    if isinstance(model_input, str):
        # Convert string to tuple with defaults
        return (model_input, "main", None, False)
    elif isinstance(model_input, (tuple, list)) and len(model_input) >= 1:
        # Ensure tuple has all 4 elements with defaults
        model_name = model_input[0]
        revision = model_input[1] if len(model_input) > 1 else "main"
        subfolder = model_input[2] if len(model_input) > 2 else None
        _target_mode = model_input[3] if len(model_input) > 3 else False
        return (model_name, revision, subfolder, _target_mode)
    else:
        raise ValueError(f"Invalid model input format: {model_input}. Must be string or tuple.")

def get_model_tokenizer_hash_and_likes(model_input, verbose=False):
    """Get tokenizer_hash and likes for a single model. Used for parallel processing."""
    import os
    import traceback
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    from tokenizer_evaluate import calculate_vocab_hash, get_tokenizer_config, get_tokenizer_hash_field, generate_model_key
    
    try:
        # Normalize input to tuple format
        model_name, revision, subfolder, _target_mode = normalize_model_input(model_input)
        
        # Generate model key for consistent identification
        model_key = generate_model_key(model_name, revision, subfolder, _target_mode)
        
        # Get likes - first check cache, then API
        if model_name in MODEL_NAME_TO_LIKES:
            likes = MODEL_NAME_TO_LIKES[model_name]
            created_at = MODEL_NAME_TO_CREATED_AT.get(model_name, datetime.datetime.now().isoformat())
            if verbose:
                print(f"Using cached likes for {model_name}: {likes}")
        else:
            # Fall back to HF API
            try:
                api = HfApi()
                model_info = api.model_info(model_name)
                likes = model_info.likes
                # Normalize created_at to ISO format string
                if hasattr(model_info.created_at, 'isoformat'):
                    created_at = model_info.created_at.isoformat()
                else:
                    created_at = str(model_info.created_at)
                
                # Cache the fetched data
                MODEL_NAME_TO_LIKES[model_name] = likes
                MODEL_NAME_TO_CREATED_AT[model_name] = created_at
                save_likes_and_created_at_cache()  # Save immediately to preserve data
                
                if verbose:
                    print(f"Fetched likes from API for {model_name}: {likes}")
            except Exception as e:
                if verbose:
                    print(f"Could not fetch likes for {model_name}: {e}")
                likes = 0
                created_at = datetime.datetime.now().isoformat()
        
        # Get tokenizer hash
        tokenizer_hash_field = get_tokenizer_hash_field()
        if model_key not in MODEL_NAME_TO_TOKENIZER_HASHES:
            kwargs = {
                'trust_remote_code': True,
                'revision': revision,
            }
            if subfolder:
                kwargs['subfolder'] = subfolder
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_fast=True, **kwargs
                )
            except Exception as e:
                #if verbose:
                #    print(f"Error loading tokenizer {model_name} with use_fast=True, trying without it: {e}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, **kwargs
                )
        
            # Apply target mode if requested - this must be done before calculating hash
            if _target_mode:
                if hasattr(tokenizer, '_switch_to_target_mode'):
                    try:
                        if verbose:
                            print(f"Switching tokenizer to target mode for {model_name}...")
                        tokenizer._switch_to_target_mode()
                        if verbose:
                            print(f"Successfully switched tokenizer to target mode for {model_name}")
                    except Exception as e:
                        if verbose:
                            print(f"Failed to switch tokenizer to target mode for {model_name}: {e}")
                elif hasattr(tokenizer, 'as_target_tokenizer'):
                    try:
                        if verbose:
                            print(f"Switching tokenizer to target mode via as_target_tokenizer for {model_name}...")
                        tokenizer.as_target_tokenizer()
                        if verbose:
                            print(f"Successfully switched tokenizer to target mode for {model_name}")
                    except Exception as e:
                        if verbose:
                            print(f"Failed to switch tokenizer to target mode for {model_name}: {e}")
                elif verbose:
                    print(f"Tokenizer {model_name} does not support target mode switching")
        
            # Calculate both vocab hash and tokenizer config hash, then use the appropriate one
            vocab_hash, _ = calculate_vocab_hash(tokenizer)
            
            if tokenizer_hash_field == 'tokenizer_config_hash':
                _, tokenizer_config_hash = get_tokenizer_config(tokenizer, vocab_hash)
                tokenizer_hash = tokenizer_config_hash
            else:
                tokenizer_hash = vocab_hash
                
            MODEL_NAME_TO_TOKENIZER_HASHES[model_key] = tokenizer_hash
        else:
            tokenizer_hash = MODEL_NAME_TO_TOKENIZER_HASHES[model_key]
        
        return {
            'model_name': model_name,
            'revision': revision,
            'subfolder': subfolder,
            '_target_mode': _target_mode,
            'model_key': model_key,
            'tokenizer_hash': tokenizer_hash,
            'likes': likes,
            'created_at': created_at,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        # Handle case where model_input might be invalid
        error_msg = f"Error processing {model_input}: {str(e)}"

        print(f"✗ {error_msg}")
        #traceback.print_exc()

        return {
            'model_name': model_name,
            'revision': revision,
            'subfolder': subfolder,
            '_target_mode': _target_mode,
            'model_key': model_key,
            'tokenizer_hash': None,
            'likes': 0,
            'created_at': datetime.datetime.now().isoformat(),
            'success': False,
            'error': error_msg
        }

def deduplicate_candidates_by_tokenizer_hash(candidate_results, verbose=False):
    """Deduplicate candidates keeping the one with most likes for each tokenizer_hash."""
    tokenizer_hash_to_best = {}  # tokenizer_hash -> candidate_result
    
    for candidate in candidate_results:
        if not candidate['success'] or not candidate['tokenizer_hash']:
            continue
            
        tokenizer_hash = candidate['tokenizer_hash']
        likes = candidate['likes']
        created_at = candidate['created_at']
        model_key = candidate['model_key']
        
        if tokenizer_hash not in tokenizer_hash_to_best or likes > tokenizer_hash_to_best[tokenizer_hash]['likes'] or (likes == tokenizer_hash_to_best[tokenizer_hash]['likes'] and created_at < tokenizer_hash_to_best[tokenizer_hash]['created_at']):
            tokenizer_hash_to_best[tokenizer_hash] = candidate
    
    dedup_candidates = list(tokenizer_hash_to_best.values())
    
    if verbose:
        print(f"Deduplicated {len(candidate_results)} candidates to {len(dedup_candidates)} unique tokenizer_hashes")
        
    return dedup_candidates

def safe_compare_created_at(created_at1, created_at2):
    """Safely compare two created_at values, handling both strings and datetime objects."""
    # Normalize both to strings for comparison
    def normalize_created_at(created_at):
        if created_at is None:
            return datetime.datetime.now().isoformat()
        if hasattr(created_at, 'isoformat'):
            return created_at.isoformat()
        return str(created_at)
    
    created_at1_str = normalize_created_at(created_at1)
    created_at2_str = normalize_created_at(created_at2)
    
    # Compare as strings (ISO format strings compare correctly lexicographically)
    return created_at1_str < created_at2_str

def filter_candidates_vs_evaluated(dedup_candidates, evaluated_tokenizer_hashes, evaluated_model_keys, verbose=False):
    """Filter candidates to only include those that should be evaluated."""
    models_to_evaluate = []
    skipped_models = []
    
    for candidate in dedup_candidates:
        tokenizer_hash = candidate['tokenizer_hash']
        likes = candidate['likes']
        model_name = candidate['model_name']
        model_key = candidate['model_key']
        created_at = candidate['created_at']
        
        should_evaluate = False
        reason = ""
        
        # First check if this exact model_key has already been evaluated
        if model_key in evaluated_model_keys:
            should_evaluate = False
            reason = "exact model configuration already evaluated"
        elif tokenizer_hash not in evaluated_tokenizer_hashes:
            # New tokenizer_hash, should evaluate
            should_evaluate = True
            reason = "new tokenizer_hash"
        else:
            # Tokenizer_hash exists, check if this model has more likes
            evaluated_model_key, evaluated_likes, evaluated_created_at = evaluated_tokenizer_hashes[tokenizer_hash]
            if likes > evaluated_likes:
                should_evaluate = True
                reason = f"more likes ({likes} > {evaluated_likes})"
            elif likes == evaluated_likes and safe_compare_created_at(created_at, evaluated_created_at):
                should_evaluate = True
                reason = f"older model ({created_at} < {evaluated_created_at})"
            else:
                reason = f"fewer likes and/or newer model ({likes} <= {evaluated_likes}, {created_at} >= {evaluated_created_at}, current best: {evaluated_model_key})"
        
        if should_evaluate:
            models_to_evaluate.append(candidate)
            if verbose:
                print(f"  ✓ {model_key}: {reason}")
        else:
            skipped_models.append((model_key, reason))
            if verbose:
                print(f"  ✗ {model_key}: {reason}")
    
    if verbose:
        print(f"Selected {len(models_to_evaluate)} models to evaluate, skipped {len(skipped_models)}")
        
    return models_to_evaluate, skipped_models

def get_processed_models(results_dir, verbose=False):
    """Check which models have already been processed by looking at result files."""
    from pathlib import Path
    import json
    
    results_path = Path(results_dir) / "evaluations"
    if not results_path.exists():
        return set()
    
    processed_models = set()
    
    for result_file in results_path.glob("results_*.jsonl"):
        try:
            # Read the first line of the JSONL file to get the model name
            with open(result_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    data = json.loads(first_line)
                    if 'model' in data:
                        processed_models.add(data['model'])
        except Exception as e:
            # If we can't read the file, skip it
            if verbose:
                print(f"Warning: Could not read {result_file}: {e}")
            continue
            
    return processed_models

def smart_model_selection(all_tasks, results_dir, num_processes=1, verbose=False):
    """
    Implement smart model selection based on tokenizer_hash and likes.
    
    Returns:
        filtered_tasks: List of tasks that should be evaluated
        skipped_info: Dict with information about skipped models
    """
    import multiprocessing as mp
    import time
    from functools import partial
    from tqdm import tqdm
    
    print("=== Starting Smart Model Selection ===")
    print("This process will:")
    print("  1. Extract tokenizer_hash and likes from already evaluated models")
    print("  2. Calculate tokenizer_hash and likes for all candidate models")
    print("  3. Deduplicate candidates by tokenizer_hash (keeping highest likes)")
    print("  4. Only select models with new tokenizer_hash or higher likes than existing")
    print()
    
    # Step 1: Get tokenizer_hashes and likes from already evaluated models
    print("1. Loading already evaluated models...")
    evaluated_tokenizer_hashes, all_model_names, all_model_keys = get_evaluated_tokenizer_hashes_and_likes(results_dir, verbose)
    print(f"   Found {len(evaluated_tokenizer_hashes)} unique tokenizer_hashes in evaluated models")
    print(f"   Found {len(all_model_names)} unique model names in evaluated models")
    print(f"   Found {len(all_model_keys)} unique model keys in evaluated models")
    
    # Step 2: Extract candidate models from tasks
    candidate_models = []
    for task in all_tasks:
        model_input = task[0]  # First element is the model configuration
        candidate_models.append(model_input)
    
    print(f"2. Processing {len(candidate_models)} candidate models...")

    #Step 2.1: Dedup task list by model_key
    already_seen_model_keys = set()
    filtered_tasks_dedup = []
    for task in all_tasks:
        model_input = task[0]
        try:
            model_name, revision, subfolder, _target_mode = normalize_model_input(model_input)

            if model_name in BLACKLIST_MODEL_NAMES:
                continue
            
            from tokenizer_evaluate import generate_model_key
            model_key = generate_model_key(model_name, revision, subfolder, _target_mode)
            if model_key in already_seen_model_keys:
                continue
            already_seen_model_keys.add(model_key)
            filtered_tasks_dedup.append(task)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not normalize model input {model_input}: {e}")
            continue
    
    print(f"   Deduplicated {len(all_tasks)} task list to {len(filtered_tasks_dedup)} candidate models")
    all_tasks = filtered_tasks_dedup
    candidate_models = [task[0] for task in all_tasks]
    
    #Step 2.2 Filter out already evaluated models by model_key
    old_count = len(candidate_models)
    filtered_models = []
    filtered_tasks_by_eval = []
    
    for i, model_input in enumerate(candidate_models):
        try:
            model_name, revision, subfolder, _target_mode = normalize_model_input(model_input)
            from tokenizer_evaluate import generate_model_key
            model_key = generate_model_key(model_name, revision, subfolder, _target_mode)
            
            # Check if this specific model configuration was already evaluated using exact model_key match
            if model_key not in all_model_keys:
                filtered_models.append(model_input)
                filtered_tasks_by_eval.append(all_tasks[i])
        except Exception as e:
            if verbose:
                print(f"Warning: Could not check evaluation status for {model_input}: {e}")
            continue
    
    candidate_models = filtered_models
    all_tasks = filtered_tasks_by_eval
    print(f"   Filtered out {old_count - len(candidate_models)} already evaluated models by exact model_key match")
    print(f"   {len(candidate_models)} candidate models left")

    # Step 3: Get tokenizer_hash and likes for all candidates in parallel
    print("3. Calculating tokenizer_hash and likes for candidates...")
    start_time = time.time()
    
    if num_processes <= 1:
        # Single process
        candidate_results = []
        for model_input in tqdm(candidate_models, desc="   🔍 Analyzing candidate models", unit="model"):
            if verbose:
                print(f"Processing {model_input}")
            result = get_model_tokenizer_hash_and_likes(model_input, verbose)
            candidate_results.append(result)
    else:
        # Multiprocess
        # Use functools.partial to create a picklable function
        process_model_func = partial(get_model_tokenizer_hash_and_likes, verbose=verbose)
        
        with mp.Pool(processes=num_processes) as pool:
            # Use imap with tqdm for progress tracking
            candidate_results = []
            with tqdm(total=len(candidate_models), desc="   🔍 Analyzing candidate models", unit="model") as pbar:
                for result in pool.imap(process_model_func, candidate_models):
                    candidate_results.append(result)
                    pbar.update(1)

    # Update cache with results
    for r in candidate_results:
        if r['success'] and r['model_key'] and r['tokenizer_hash']:
            MODEL_NAME_TO_TOKENIZER_HASHES[r['model_key']] = r['tokenizer_hash']
    
    # Save caches
    with open(TEMP_MODEL_NAME_TO_TOKENIZER_HASHES_FILE, 'w', encoding='utf-8') as f:
        json.dump(MODEL_NAME_TO_TOKENIZER_HASHES, f, ensure_ascii=False, indent=4)
    
    save_likes_and_created_at_cache()
    
    processing_time = time.time() - start_time
    successful_candidates = [r for r in candidate_results if r['success']]
    failed_candidates = [r for r in candidate_results if not r['success']]
    
    print(f"   Processed {len(successful_candidates)}/{len(candidate_models)} successfully in {processing_time:.1f}s")
    if failed_candidates and verbose:
        print(f"   Failed models: {[r['model_key'] for r in failed_candidates[:5]]}" + 
              (f" and {len(failed_candidates)-5} more" if len(failed_candidates) > 5 else ""))
    
    # Step 4: Deduplicate candidates by tokenizer_hash (keep highest likes)
    print("4. Deduplicating candidates by tokenizer_hash...")
    dedup_candidates = deduplicate_candidates_by_tokenizer_hash(successful_candidates, verbose)
    
    # Step 5: Compare with evaluated models and filter
    print("5. Comparing with evaluated models...")
    models_to_evaluate, skipped_models = filter_candidates_vs_evaluated(
        dedup_candidates, evaluated_tokenizer_hashes, all_model_keys, verbose
    )
    
    # Step 6: Map back to original tasks using model_key
    models_to_evaluate_keys = {m['model_key'] for m in models_to_evaluate}
    filtered_tasks = []
    
    for task in all_tasks:
        model_input = task[0]
        try:
            model_name, revision, subfolder, _target_mode = normalize_model_input(model_input)
            from tokenizer_evaluate import generate_model_key
            model_key = generate_model_key(model_name, revision, subfolder, _target_mode)
            if model_key in models_to_evaluate_keys:
                filtered_tasks.append(task)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not map task back for {model_input}: {e}")
            continue
    
    print(f"=== Smart Selection Complete ===")
    print(f"Final result: {len(filtered_tasks)}/{len(all_tasks)} models selected for evaluation")
    
    skipped_info = {
        'by_tokenizer_hash_dedup': len(successful_candidates) - len(dedup_candidates),
        'by_comparison': len(skipped_models),
        'failed_to_process': len(failed_candidates),
        'details': {
            'skipped_models': skipped_models,
            'failed_models': [r['model_key'] for r in failed_candidates]
        }
    }
    
    return filtered_tasks, skipped_info


lang_specific_models = {
    'multilingual': []
}

"""
with open('data/models_to_eval.txt', 'r') as f:
    for line in f:
        model = line.strip()
        lang_specific_models['multilingual'].append(model)
with open('data/models_to_eval_2.txt', 'r') as f:
    for line in f:
        model = line.strip()
        lang_specific_models['multilingual'].append(model)

with open('data/models_to_eval_3.txt', 'r') as f:
    for line in f:
        model = line.strip()
        lang_specific_models['multilingual'].append(model) 


import json
with open('data/models_to_eval_4.json', 'r') as f:
    m_list = json.load(f)
    for model in m_list:
        #if model[2] is None:
        lang_specific_models['multilingual'].append(model) 

with open('data/models_to_eval_5.txt', 'r') as f:
    for line in f:
        model = line.strip()
        lang_specific_models['multilingual'].append(model) 

import json
with open('data/models_to_eval_6.json', 'r') as f:
    m_list = json.load(f)
    for model in m_list:
        #if model[2] is None:
        lang_specific_models['multilingual'].append(model) 
            
import json
with open('data/models_to_eval_7.json', 'r') as f:
    m_list = json.load(f)
    for model in m_list:
        #if model[2] is None:
        lang_specific_models['multilingual'].append(model) 

with open('data/models_to_eval_8.txt', 'r') as f:
    for line in f:
        model = line.strip()
        lang_specific_models['multilingual'].append(model)

with open('data/models_to_eval_8_fix.txt', 'r') as f:
    for line in f:
        model = line.strip()
        lang_specific_models['multilingual'].append(model)
"""

with open('data/models_to_eval_2_1.txt', 'r') as f:
    for line in f:
        model = line.strip()
        if model not in lang_specific_models['multilingual']:
            lang_specific_models['multilingual'].append(model)

with open('data/models_to_eval_2_2.txt', 'r') as f:
    for line in f:
        model = line.strip()
        if model not in lang_specific_models['multilingual']:
            lang_specific_models['multilingual'].append(model)

if os.path.exists('data/models_to_search.jsonl'):
    with open('data/models_to_search.jsonl', 'r') as f:
        for line in f:
            model = json.loads(line)
            if model['id'] in lang_specific_models['multilingual']:
                MODEL_NAME_TO_LIKES[model['id']] = model['likes']
                MODEL_NAME_TO_CREATED_AT[model['id']] = model['created_at']

if os.path.exists('data/models_to_search_3.jsonl'):
    with open('data/models_to_search_3.jsonl', 'r') as f:
        for line in f:
            model = json.loads(line)
            if model['id'] in lang_specific_models['multilingual']:
                MODEL_NAME_TO_LIKES[model['id']] = model['likes']
                MODEL_NAME_TO_CREATED_AT[model['id']] = model['created_at']

if os.path.exists('data/models_to_search_4.jsonl'):
    with open('data/models_to_search_4.jsonl', 'r') as f:
        for line in f:
            model = json.loads(line)
            if model['id'] in lang_specific_models['multilingual']:
                MODEL_NAME_TO_LIKES[model['id']] = model['likes']
                MODEL_NAME_TO_CREATED_AT[model['id']] = model['created_at']

if os.path.exists('data/models_to_search_5.jsonl'):
    with open('data/models_to_search_5.jsonl', 'r') as f:
        for line in f:
            model = json.loads(line)
            if model['id'] in lang_specific_models['multilingual']:
                MODEL_NAME_TO_LIKES[model['id']] = model['likes']
                MODEL_NAME_TO_CREATED_AT[model['id']] = model['created_at']

if os.path.exists('data/models_to_search_6.jsonl'):
    with open('data/models_to_search_6.jsonl', 'r') as f:
        for line in f:
            model = json.loads(line)
            if model['id'] in lang_specific_models['multilingual']:
                MODEL_NAME_TO_LIKES[model['id']] = model['likes']
                MODEL_NAME_TO_CREATED_AT[model['id']] = model['created_at']
                
if os.path.exists('data/models_to_search_7.jsonl'):
    with open('data/models_to_search_7.jsonl', 'r') as f:
        for line in f:
            model = json.loads(line)
            if model['id'] in lang_specific_models['multilingual']:
                MODEL_NAME_TO_LIKES[model['id']] = model['likes']
                MODEL_NAME_TO_CREATED_AT[model['id']] = model['created_at']

if os.path.exists('data/models_to_search_8.jsonl'):
    with open('data/models_to_search_8.jsonl', 'r') as f:
        for line in f:
            model = json.loads(line)
            if model['id'] in lang_specific_models['multilingual']:
                MODEL_NAME_TO_LIKES[model['id']] = model['likes']
                MODEL_NAME_TO_CREATED_AT[model['id']] = model['created_at']

if os.path.exists('data/models_to_search_2.jsonl'):
    with open('data/models_to_search_2.jsonl', 'r') as f:
        for line in f:
            model = json.loads(line)
            if model['id'] in lang_specific_models['multilingual']:
                MODEL_NAME_TO_LIKES[model['id']] = model['likes']
                MODEL_NAME_TO_CREATED_AT[model['id']] = model['created_at']

if os.path.exists('data/2_models_to_search.jsonl'):
    with open('data/models_to_search_2.jsonl', 'r') as f:
        for line in f:
            model = json.loads(line)
            if model['id'] in lang_specific_models['multilingual']:
                MODEL_NAME_TO_LIKES[model['id']] = model['likes']
                MODEL_NAME_TO_CREATED_AT[model['id']] = model['created_at']


# Save the updated cache after loading from JSONL files
save_likes_and_created_at_cache()

if os.path.exists(TEMP_MODEL_NAME_TO_TOKENIZER_HASHES_FILE):
    with open(TEMP_MODEL_NAME_TO_TOKENIZER_HASHES_FILE, 'r', encoding='utf-8') as f:
        MODEL_NAME_TO_TOKENIZER_HASHES = json.load(f)

if os.path.exists(TEMP_MODEL_LIKES_AND_CREATED_AT_FILE):
    with open(TEMP_MODEL_LIKES_AND_CREATED_AT_FILE, 'r', encoding='utf-8') as f:
        likes_and_created_at_data = json.load(f)
        cached_likes = likes_and_created_at_data.get('likes', {})
        cached_created_at = likes_and_created_at_data.get('created_at', {})
        MODEL_NAME_TO_LIKES.update(cached_likes)
        MODEL_NAME_TO_CREATED_AT.update(cached_created_at)
        print(f"Loaded {len(cached_likes)} cached likes and {len(cached_created_at)} cached created_at values")

if os.path.exists(BLACKLIST_FILE):
    with open(BLACKLIST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            BLACKLIST_MODEL_NAMES.add(line.strip())


print(len(MODEL_NAME_TO_LIKES))
print(len(MODEL_NAME_TO_CREATED_AT))


"""
already_added_models = set()
for _, models in lang_specific_models.items():
    for model in models:
        if isinstance(model, str):
            already_added_models.add(model)
for model in GEOTREND_MODELS + HPLT_MODELS + helsinki_models + kochi_yasuo_models:
    if model not in already_added_models:
        lang_specific_models['multilingual'].append(model)


#add target mode for Helsinki-NLP models
for _, models in lang_specific_models.items():
    for model in models.copy():
        if 'Helsinki-NLP' in model:
            models.append((model, 'main', None, True))
"""
if __name__ == "__main__":
    import argparse

    # Simple argument parsing
    parser = argparse.ArgumentParser(
        description='Run benchmarks on all language-specific models with smart deduplication',
        epilog='''
Examples:
  # Use smart selection to avoid duplicate tokenizers
  python init_lb.py --smart-selection --verbose
  
  # Use smart selection with parallel processing
  python init_lb.py --smart-selection --selection-processes 8 --num-processes 4
  
  # Force re-evaluation of all models
  python init_lb.py --force --num-processes 2
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--num-processes', type=int, default=1, help='Number of processes for evaluation (default: 1)')
    parser.add_argument('--force', action='store_true', help='Force re-evaluation of all models')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--smart-selection', action='store_true', help='Use smart model selection: deduplicate by tokenizer_hash and select models with highest likes')
    parser.add_argument('--selection-processes', type=int, default=4, help='Number of processes for tokenizer_hash calculation in smart selection (default: 4)')
    
    args = parser.parse_args()
    
    import os
    import multiprocessing as mp
    import sys
    import time
    from functools import partial
    import traceback
    from tqdm import tqdm
    

    #disable tokenizer paralleism
    if args.num_processes > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    #set donwload directory
    #os.environ["HF_HOME"] = os.path.abspath("data/tokenizers")

    from tokenizer_evaluate import run_benchmark, download_repos_once
    from pathlib import Path



    def run_model_benchmark(model_info, benchmark_dir, benchmark_file_hashes, tokenizer_hash_cache, force, verbose, progress_info):
        """Worker function to run benchmark on a single model."""
        model_input, lang_key, languages_to_eval = model_info
        current, total = progress_info
        
        try:
            # Normalize model input to extract all parameters
            model_name, revision, subfolder, _target_mode = normalize_model_input(model_input)
            
            print(f"[{os.getpid()}] [{current}/{total}] Starting {model_name} (revision: {revision}, subfolder: {subfolder}, target_mode: {_target_mode}) ({lang_key})")
            
            # Parse languages string into list if needed
            langs_list = None
            if isinstance(languages_to_eval, str) and languages_to_eval:
                langs_list = [lang.strip() for lang in languages_to_eval.split(',') if lang.strip()]
                
            df = run_benchmark(
                model_name=model_name,
                revision=revision,
                subfolder=subfolder,
                _target_mode=_target_mode,
                benchmark_dir=benchmark_dir,
                langs=langs_list,
                verbose=verbose,
                save_results=True,
                force_rerun=force,
                benchmark_file_hashes=benchmark_file_hashes,
                tokenizer_hash_cache=tokenizer_hash_cache,
                upload_results=False,
                trust_remote_code=True,
                use_batched_evaluation=True,
                batch_size=1000
            )
            
            print(f"[{os.getpid()}] [{current}/{total}] ✓ Completed {model_name}")
            return {'model_input': model_input, 'model_name': model_name, 'success': True, 'error': None}
            
        except Exception as e:
            error_msg = f"Error processing {model_input}: {str(e)}"
            print(f"[{os.getpid()}] [{current}/{total}] ✗ {error_msg}")
            if verbose:
                traceback.print_exc()
            return {'model_input': model_input, 'model_name': getattr(model_input, '__str__', lambda: str(model_input))(), 'success': False, 'error': error_msg}

    # Download repos
    print("Setting up repositories...")
    benchmark_dir, results_dir, benchmark_file_hashes, tokenizer_hash_cache = download_repos_once()
    
    # Build initial task list from all models
    all_tasks = []
    for lang_key, models in lang_specific_models.items():
        languages_to_eval = model_lang_eval_mapping.get(lang_key, None)
        for model_input in models:
            # Normalize model input to ensure consistent format
            try:
                model_name, revision, subfolder, _target_mode = normalize_model_input(model_input)
                # Create task with normalized model information
                all_tasks.append((model_input, lang_key, languages_to_eval))
            except Exception as e:
                print(f"Warning: Skipping invalid model input {model_input}: {e}")
                continue
    
    print(f"Found {len(all_tasks)} total models")
    
    # Select models to process
    if args.smart_selection and not args.force:
        # Use smart selection based on tokenizer_hash and likes
        tasks, skipped_info = smart_model_selection(
            all_tasks, 
            results_dir, 
            num_processes=args.selection_processes, 
            verbose=args.verbose
        )
        
        print(f"\nSmart selection summary:")
        print(f"  - Deduplicated: {skipped_info['by_tokenizer_hash_dedup']} models")
        print(f"  - Skipped (lower likes and/or newer model): {skipped_info['by_comparison']} models")
        print(f"  - Failed to process: {skipped_info['failed_to_process']} models")
        
        if args.verbose and skipped_info['details']['skipped_models']:
            print(f"\nSkipped models (first 10):")
            for model_name, reason in skipped_info['details']['skipped_models'][:10]:
                print(f"  ✗ {model_name}: {reason}")
            if len(skipped_info['details']['skipped_models']) > 10:
                print(f"  ... and {len(skipped_info['details']['skipped_models']) - 10} more")
    
        """
        for task in tasks:
            model_input = task[0]
            model_name, revision, subfolder, _target_mode = normalize_model_input(model_input)
            BLACKLIST_MODEL_NAMES.add(model_name)
        with open(BLACKLIST_FILE, 'w', encoding='utf8') as f:
            for m in BLACKLIST_MODEL_NAMES:
                f.write(m + '\n')
        """
    elif not args.force:
        # Use simple selection (existing logic)
        print("Checking for already processed models...")
        processed_models = get_processed_models(results_dir, args.verbose)
        print(f"Found {len(processed_models)} already processed models")
        
        # Filter out already processed models
        tasks = []
        skipped = []
        for task in all_tasks:
            model_input = task[0]
            try:
                # Extract model_name from input (whether string or tuple)
                model_name, revision, subfolder, _target_mode = normalize_model_input(model_input)
                if model_name in processed_models:
                    skipped.append(str(model_input))
                else:
                    tasks.append(task)
            except Exception as e:
                print(f"Warning: Could not process model input {model_input}: {e}")
                # Default to string comparison for backwards compatibility
                if model_input in processed_models:
                    skipped.append(str(model_input))
                else:
                    tasks.append(task)
        
        if skipped:
            print(f"Skipping {len(skipped)} already processed models")
            if args.verbose:
                for model in skipped[:5]:  # Show first 5
                    print(f"  ✓ {model}")
                if len(skipped) > 5:
                    print(f"  ... and {len(skipped) - 5} more")
    else:
        tasks = all_tasks
        print("Force mode: will re-process all models")
    
    if not tasks:
        print("All models already processed! Use --force to re-process.")
        sys.exit(0)
    
    print(f"Processing {len(tasks)} models with {args.num_processes} processes...")
    
    start_time = time.time()
    
    if args.num_processes <= 1:
        # Single process
        results = []
        for i, task in enumerate(tqdm(tasks, desc="🚀 Running tokenizer benchmarks", unit="model"), 1):
            print(f"\n=== Task {i}/{len(tasks)}: {task[0]} ===")
            result = run_model_benchmark(task, benchmark_dir, benchmark_file_hashes, tokenizer_hash_cache, args.force, args.verbose, (i, len(tasks)))
            results.append(result)
    else:
        # Multiprocess - need to add progress info to each task
        tasks_with_progress = [(task, (i+1, len(tasks))) for i, task in enumerate(tasks)]
        
        def worker_wrapper(task_with_progress):
            task, progress_info = task_with_progress
            return run_model_benchmark(task, benchmark_dir, benchmark_file_hashes, tokenizer_hash_cache, args.force, args.verbose, progress_info)
        
        with mp.Pool(processes=args.num_processes) as pool:
            # Use imap with tqdm for progress tracking
            results = []
            with tqdm(total=len(tasks_with_progress), desc="🚀 Running tokenizer benchmarks", unit="model") as pbar:
                for result in pool.imap(worker_wrapper, tasks_with_progress):
                    results.append(result)
                    pbar.update(1)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"COMPLETED: {successful}/{len(results)} successful")
    print(f"Time: {total_time:.1f}s ({total_time/len(results):.1f}s avg)")
    print(f"{'='*50}")
    
    if failed > 0:
        print("\nFailed models:")
        for result in results:
            if not result['success']:
                print(f"  - {result['model_name']}: {result['error']}")

