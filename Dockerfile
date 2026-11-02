FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app app/ ./app/
COPY --chown=app:app src/ ./src/
COPY --chown=app:app models/ ./models/
COPY --chown=app:app data/sample_orders_batch.csv ./data/sample_orders_batch.csv
COPY --chown=app:app reports/figures/ ./reports/figures/

USER app

EXPOSE 5000

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} app.app:app"]
