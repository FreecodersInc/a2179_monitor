# EXP-027 inventory

Cap: 50 MiB. Sizes are Hub metadata (needed files), not a download.

## Machine

- nvidia-smi: missing or failed (nvidia-smi not found; torch not importable: ModuleNotFoundError: No module named 'torch')
- torch: not importable, cuda_available=False, device=—
- disk free: 101007.8 MiB, RAM available: 11526.5 MiB

## Models

| id | needed MiB | primary | fits 50 MiB | note |
|---|---:|---|---|---|
| `cross-encoder/ms-marco-TinyBERT-L2-v2` | 17.6 | model.safetensors (16.7 MiB) | yes | gated=False |
| `cross-encoder/ms-marco-TinyBERT-L4-v2` | – | – | no | http 401 |
| `cross-encoder/ms-marco-MiniLM-L-2-v2` | 60.5 | model.safetensors (59.6 MiB) | no | gated=False |
| `cross-encoder/ms-marco-MiniLM-L-4-v2` | 74.0 | model.safetensors (73.1 MiB) | no | gated=False |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 87.6 | model.safetensors (86.7 MiB) | no | gated=False |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 128.2 | model.safetensors (127.3 MiB) | no | gated=False |
| `cross-encoder/stsb-TinyBERT-L4` | 55.7 | model.safetensors (54.8 MiB) | no | gated=False |
| `cross-encoder/nli-MiniLM2-L6-H768` | 315.8 | model.safetensors (313.3 MiB) | no | gated=False |
| `cross-encoder/nli-distilroberta-base` | 317.9 | model.safetensors (313.3 MiB) | no | gated=False |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 469.9 | model.safetensors (448.8 MiB) | no | gated=False |
| `cross-encoder/mmarco-mMiniLMv2-L6-H384-v1` | – | – | no | http 401 |
| `BAAI/bge-reranker-base` | 1081.8 | model.safetensors (1060.7 MiB) | no | gated=False |
| `BAAI/bge-reranker-large` | 2157.0 | model.safetensors (2135.9 MiB) | no | gated=False |
| `BAAI/bge-reranker-v2-m3` | 2187.0 | model.safetensors (2165.9 MiB) | no | gated=False |
| `jinaai/jina-reranker-v1-tiny-en` | 66.1 | model.safetensors (63.0 MiB) | no | gated=False |
| `jinaai/jina-reranker-v1-turbo-en` | 75.1 | model.safetensors (72.1 MiB) | no | gated=False |
| `jinaai/jina-reranker-v2-base-multilingual` | 547.4 | model.safetensors (531.1 MiB) | no | gated=False |
| `mixedbread-ai/mxbai-rerank-xsmall-v1` | 143.4 | model.safetensors (135.1 MiB) | no | gated=False |
| `mixedbread-ai/mxbai-rerank-base-v1` | 360.0 | model.safetensors (351.8 MiB) | no | gated=False |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 599.9 | model.safetensors (583.6 MiB) | no | gated=False |
| `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | 429.3 | model.safetensors (408.2 MiB) | no | gated=False |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | 547.4 | model.safetensors (531.8 MiB) | no | gated=False |
| `typeform/distilbert-base-uncased-mnli` | 255.6 | model.safetensors (255.4 MiB) | no | gated=False |
| `colbert-ir/colbertv2.0` | 418.7 | model.safetensors (418.0 MiB) | no | gated=False |
| `answerdotai/answerai-colbert-small-v1` | 128.3 | model.safetensors (127.4 MiB) | no | gated=False |
| `lightonai/GTE-ModernColBERT-v1` | 571.9 | model.safetensors (568.5 MiB) | no | gated=False |
| `naver/splade-cocondenser-ensembledistil` | 418.5 | pytorch_model.bin (417.8 MiB) | no | gated=False |
| `naver/splade-v3` | 418.7 | pytorch_model.bin (417.8 MiB) | no | gated=auto |
| `naver/splade-v3-distilbert` | 256.5 | pytorch_model.bin (255.6 MiB) | no | gated=False |
| `naver/splade-v3-lexical` | 418.7 | pytorch_model.bin (417.8 MiB) | no | gated=auto |
| `naver/splade_v2_distil` | 255.8 | pytorch_model.bin (255.6 MiB) | no | gated=False |
| `naver/efficient-splade-VI-BT-large-query` | 17.5 | pytorch_model.bin (16.9 MiB) | yes | gated=False |
| `naver/efficient-splade-VI-BT-large-doc` | 256.2 | pytorch_model.bin (255.6 MiB) | no | gated=False |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 462.3 | model.safetensors (448.8 MiB) | no | gated=False |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 1074.2 | model.safetensors (1060.7 MiB) | no | gated=False |
| `intfloat/multilingual-e5-small` | 470.0 | model.safetensors (448.8 MiB) | no | gated=False |
| `intfloat/multilingual-e5-base` | 1081.8 | model.safetensors (1060.7 MiB) | no | gated=False |
| `intfloat/multilingual-e5-large` | 2157.0 | model.safetensors (2135.9 MiB) | no | gated=False |
| `intfloat/multilingual-e5-large-instruct` | 1089.1 | model.safetensors (1067.9 MiB) | no | gated=False |
| `BAAI/bge-m3` | 2187.1 | pytorch_model.bin (2165.9 MiB) | no | gated=False |
| `sentence-transformers/LaBSE` | 1812.9 | model.safetensors (1796.5 MiB) | no | gated=False |
| `BAAI/bge-base-en-v1.5` | 418.6 | model.safetensors (417.7 MiB) | no | gated=False |
| `intfloat/e5-base-v2` | 418.6 | model.safetensors (417.7 MiB) | no | gated=False |
| `sentence-transformers/all-MiniLM-L6-v2` | 87.3 | model.safetensors (86.7 MiB) | no | gated=False |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | 67.0 | model.safetensors (66.3 MiB) | no | gated=False |
| `sentence-transformers/paraphrase-MiniLM-L6-v2` | 87.3 | model.safetensors (86.7 MiB) | no | gated=False |
| `sentence-transformers/paraphrase-albert-small-v2` | 46.6 | model.safetensors (44.6 MiB) | yes | gated=False |
| `TaylorAI/bge-micro-v2` | 34.1 | model.safetensors (33.2 MiB) | yes | gated=False |
| `TaylorAI/gte-tiny` | 44.3 | model.safetensors (43.4 MiB) | yes | gated=False |
| `thenlper/gte-small` | 64.6 | model.safetensors (63.7 MiB) | no | gated=False |
| `BAAI/bge-small-en-v1.5` | 128.2 | model.safetensors (127.3 MiB) | no | gated=False |
| `intfloat/e5-small-v2` | 128.2 | model.safetensors (127.3 MiB) | no | gated=False |
| `cointegrated/rubert-tiny2` | 114.8 | model.safetensors (112.1 MiB) | no | gated=False |
| `nreimers/mMiniLMv2-L6-H384` | – | – | no | http 401 |
| `microsoft/Multilingual-MiniLM-L12-H384` | 453.7 | pytorch_model.bin (448.9 MiB) | no | gated=False |
