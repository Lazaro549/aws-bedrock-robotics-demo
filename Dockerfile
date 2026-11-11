FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY fixtures ./fixtures

RUN pip install --no-cache-dir -e .

ENV PIPELINE_MODE=mock
ENV BRIDGE_RECORD_SCENES=/app/logs/scenes.jsonl
ENV BRIDGE_RATE_LIMIT_PER_MIN=30

EXPOSE 8787

CMD ["python", "-m", "src.bridge_server", "--host", "0.0.0.0", "--port", "8787"]
