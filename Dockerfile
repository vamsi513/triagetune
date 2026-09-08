FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY src ./src

# Replaced by the service command when serving is implemented in Phase 5.
CMD ["python", "-m", "src.inference"]
