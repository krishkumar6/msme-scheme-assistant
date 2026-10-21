FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# pinned versions first (the Chroma index format must match the version that built it)
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY eval ./eval
COPY scripts ./scripts
COPY training ./training
COPY data/raw/manifest.json ./data/raw/manifest.json

ENV NABARD_QA_DATA_DIR=/app/data NABARD_QA_RAW_DIR=/app/data/raw NABARD_QA_PROCESSED_DIR=/app/data/processed \
    NABARD_QA_INDEX_DIR=/app/data/index NABARD_QA_MODEL_CACHE_DIR=/app/data/models \
    NABARD_QA_DB_PATH=/app/data/app.db NABARD_QA_ADAPTER_PATH=/app/data/adapter/query_adapter.npz

# Prebuilt search index + query adapter (built by `nabard-qa ingest` / `train-adapter` from the 18
# source PDFs, which are too large for the repo). Set INDEX_URL= (empty) to build an image without
# it and ingest into the /app/data volume instead (see docker-compose.yml).
ARG INDEX_URL=https://github.com/krishkumar6/msme-scheme-assistant/releases/download/index-2026-10-21/index-bundle.tar.gz
ARG INDEX_SHA256=2e1a4c4a55c4c4fa5fc8f91d70a1b6c35293f6c48cf8ebb5ca621003a6bf17da
RUN if [ -n "$INDEX_URL" ]; then python scripts/fetch_index.py "$INDEX_URL" "$INDEX_SHA256" /app/data; fi

# Download the embedding and reranker models at build time, so a cold start doesn't wait on them
RUN python -c "from nabard_qa.config import get_settings; from nabard_qa.retrieval.embeddings import Embedder; \
from nabard_qa.retrieval.reranker import CrossEncoderReranker; s = get_settings(); \
Embedder(s.embedding_model, s.model_cache_dir, s.query_instruction).embed_query('warm up'); \
CrossEncoderReranker(s.reranker_model, s.model_cache_dir).score('warm up', ['warm up'])"

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app
# hosts such as Render pass the port to listen on in $PORT
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "exec nabard-qa serve --host 0.0.0.0 --port \"$PORT\""]
