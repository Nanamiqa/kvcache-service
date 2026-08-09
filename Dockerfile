FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    KVCACHE_STORE_DIR=/data/kv-cache

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" 'torch>=2.2,<3' \
    && python -m pip install --no-cache-dir '.[local]' \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/kv-cache /models/huggingface \
    && chown -R appuser:appuser /data /models

USER appuser

EXPOSE 8080
VOLUME ["/data/kv-cache", "/models/huggingface"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

CMD ["kvcache-server"]
