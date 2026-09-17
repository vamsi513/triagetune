FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRIAGETUNE_ADAPTER_DIR=/models/adapter \
    TRIAGETUNE_CACHE_DIR=/models/cache \
    TRIAGETUNE_REPORT_PATH=/app/reports/lora_adapter_test.json

WORKDIR /app

COPY requirements-serving.txt ./
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu' \
    && python -m pip install --no-cache-dir -r requirements-serving.txt

COPY api ./api
COPY src ./src
COPY reports/lora_adapter_test.json ./reports/lora_adapter_test.json

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
