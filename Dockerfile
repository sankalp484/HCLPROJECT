FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_PATH=/app/state/app.db

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY scripts ./scripts
COPY data/*.csv data/data_card.md ./data/

# run as an unprivileged user; only /app/state is writable
RUN useradd --create-home --uid 10001 app && mkdir -p /app/state && chown app /app/state
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
