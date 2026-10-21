FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY eval ./eval
COPY scripts ./scripts
COPY training ./training

# data/ (PDFs, index, models, feedback DB) is a volume so feedback and the
# index survive restarts; run `nabard-qa ingest` once against it.
VOLUME ["/app/data"]
ENV NABARD_QA_DATA_DIR=/app/data NABARD_QA_RAW_DIR=/app/data/raw NABARD_QA_PROCESSED_DIR=/app/data/processed \
    NABARD_QA_INDEX_DIR=/app/data/index NABARD_QA_MODEL_CACHE_DIR=/app/data/models \
    NABARD_QA_DB_PATH=/app/data/app.db NABARD_QA_ADAPTER_PATH=/app/data/adapter/query_adapter.npz

RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["nabard-qa", "serve", "--host", "0.0.0.0", "--port", "8000"]
