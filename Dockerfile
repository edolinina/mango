# ---------- BASE ----------
FROM python:3.13-slim AS mango-base

WORKDIR /app

# ---- Disable OpenTelemetry globally ----
ENV OTEL_SDK_DISABLED=true
ENV OTEL_TRACES_EXPORTER=none
ENV OTEL_METRICS_EXPORTER=none
ENV OTEL_LOGS_EXPORTER=none
ENV OTEL_EXPORTER_OTLP_ENDPOINT=

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# ---------- AGENT ----------
FROM mango-base AS mango-agent

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${SERVICE_PORT}"]

# ---------- MCP ----------
FROM mango-base AS mango-mcp
CMD ["python", "mcp_server/server.py"]
