PY ?= python
export PYTHONPATH := src

.PHONY: install install-models data index index-bm25 answers grade web test lint serve

install:            ## core: BM25 retrieval, evaluation, site, tests
	$(PY) -m pip install -r requirements.txt pytest ruff

install-models:     ## embeddings, re-ranker, NLI grader and the local LLM
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install sentence-transformers huggingface_hub llama-cpp-python

data:               ## download openFDA labels (about 1.5 GB) and keep the top drugs
	$(PY) scripts/fetch_labels.py

index:              ## chunk, embed, evaluate every retrieval configuration
	$(PY) -m labelrag.build

index-bm25:         ## same, keyword retrieval only (no models needed)
	$(PY) -m labelrag.build --no-dense

MODEL ?= qwen2.5-3b

answers:            ## answer the eval questions with a local LLM and grade them (MODEL=qwen2.5-1.5b|qwen2.5-3b)
	$(PY) -m labelrag.generate --model $(MODEL)

grade:              ## re-grade saved answers (MODEL=...)
	$(PY) -m labelrag.grade --tag $(MODEL)

web:                ## docs/index.html + docs/corpus.json
	$(PY) -m labelrag.web

serve: web          ## open http://localhost:8000
	$(PY) -m http.server 8000 -d docs

test:
	$(PY) -m pytest -q

lint:
	ruff check src scripts tests
