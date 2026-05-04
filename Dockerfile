# ---------- BASE ----------
FROM python:3.13-slim AS mango-base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# ---------- AGENT ----------
FROM mango-base AS mango-agent
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${SERVICE_PORT}"]

# ---------- MCP ----------
FROM mango-base AS mango-mcp
CMD ["python", "services/mcp_server.py"]