# MANGO: Managerial Agentic Network for General Orchestration

## Overview

Modern large-scale organizations continue to operate with hierarchical structures and management layers that have remained largely unchanged for decades, despite rapid advances in artificial intelligence. While many technology-driven companies have successfully integrated AI tools to support engineers and knowledge workers—primarily to accelerate coding, documentation, and analysis—these capabilities rarely extend into managerial and executive decision-making layers.

**MANGO** (Managerial Agentic Network for General Orchestration) is a research framework that explores the possibility of AI systems assuming part of the managerial workload. It investigates agentic AI architectures capable of supporting distributed, context-aware, and collaborative managerial decision-making.

---

## Key Features

- **Agentic Orchestration:** Coordinates multiple specialized AI agents, each with distinct capabilities and roles, to collaboratively solve complex tasks.
- **Central Executive (CE):** Receives high-level tasks, generates structured directives, and orchestrates the workflow by delegating subtasks to appropriate agents.
- **3-Node Reasoning Graph:** Each agent reasons via a `reason → pandas_query → validate` LangGraph cycle, grounding recommendations in real dataframe evidence before finalising.
- **Directive Management:** Each task is assigned a unique `task_id`, ensuring agent feedback and results are correctly matched to the originating directive.
- **Approval Workflow:** Supports both autonomous and human-in-the-loop modes. In non-autonomous mode, the system pauses after directive generation and presents the user with a summary for approval or rejection before proceeding.
- **Knowledge-Augmented Reasoning:** Domain knowledge from `knowledge_base/*.md` is retrieved via FAISS + HuggingFace embeddings and injected into every agent reasoning cycle.
- **Extensible Agent Design:** Agents can be easily added or configured with different capabilities, roles, and data access.

---

## System Architecture

- **Backend:**
  - Built with FastAPI.
  - Central Executive logic in `agents/central_executive.py`.
  - Per-agent reasoning in `agents/worker_network.py` (LangGraph `StateGraph`).
  - Message-passing via FastMCP (`services/mcp_server.py`).
  - REST API endpoints for task submission, approval, rejection, and results polling.

- **Frontend:**
  - React JSX UI (`static/js/app.js`), transpiled in-browser via Babel Standalone — no build step required.
  - Business-oriented design with real-time agent status cards.
  - Modal dialogs for human approval and per-agent result inspection.

- **Agents:**
  - Modular, capability-driven agents (`BusinessAgent`, `CustomerServiceAgent`, `HRAgent`).
  - Communicate via HTTP and the MCP message-passing protocol.
  - Each agent processes directives through a 3-node reasoning graph and returns structured feedback.

---

## Language Model Provider Support

MANGO supports both cloud-based and local language models:

- **OpenAI:**
  Set `LLM_PROVIDER=openai` and provide `OPENAI_API_KEY` in your `.env` file.

- **Ollama (Local LLM):**
  Set `LLM_PROVIDER=ollama` and configure `OLLAMA_URL` and `LLM_MODEL` in `.env`.

```dotenv
LLM_PROVIDER=ollama   # or 'openai'
```

---

## Configuration

All configuration is stored in `.env` at the project root. Key variables:

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=llama3
LLM_TEMPERATURE=0.2
OLLAMA_URL=http://localhost:11434
OPENAI_API_KEY=                        # required if LLM_PROVIDER=openai
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
AUTONOMOUS_MODE=false
MCP_PORT=8001
CE_PORT=8000
BUSINESS_AGENT_PORT=8010
CUSTOMERS_AGENT_PORT=8011
HR_AGENT_PORT=8012
```

**Execution Modes:**

- `AUTONOMOUS_MODE=true`: Fully automated — directives are executed immediately without user review.
- `AUTONOMOUS_MODE=false`: Human-in-the-loop mode — the system pauses for approval after directive generation.

---

## Deployment & Usage

MANGO is containerised for easy deployment using Docker Compose.

### Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed and running.

### Starting MANGO

```bash
./start_mango.sh
```

### Stopping MANGO

```bash
docker-compose down
```

### Accessing the UI

Open [http://localhost:8000](http://localhost:8000).

### Submitting a Task

1. Enter a high-level business objective in the input field and click **Run**.
2. In non-autonomous mode, review the generated directives and **Approve & Run** or **Reject**.
3. Upon approval, agents are triggered in parallel and results appear as each agent completes.
4. Click an agent card to inspect its recommendation, explanation, next steps, and validation result.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/run` | Submit a task. Returns agent list and status (`running` or `pending` for approval). |
| `POST` | `/approve` | Approve generated directives and start agent execution. |
| `POST` | `/reject` | Reject directives and cancel the workflow. |
| `GET`  | `/results` | Poll for aggregated agent feedback and results. |

---

## Agent Reasoning — 3-Node Graph

Each agent processes a directive through a `StateGraph` with three nodes:

1. **`reason`** — LLM decides the next action: `query` (need more data) or `validate` (ready to recommend). Forces validation after a maximum of 5 iterations.
2. **`pandas_query`** — Runs `create_pandas_dataframe_agent` to answer the natural language question against the agent's CSV dataset. Appends results to state, then returns to `reason`.
3. **`validate`** — Runs the pandas agent again for evidence, then asks an LLM to score the recommendation as `pass` or `fail`. On pass, builds the final `Recommendation`. On fail, returns feedback to `reason` for refinement (max 2 validation attempts).

Final output: `recommendation`, `explanation`, `next_steps`.

---

## Research Goals

MANGO explores: **Can AI systems assume part of the managerial workload and enable more adaptive, flatter organisational structures?**

By orchestrating distributed, context-aware, and collaborative AI agents, MANGO aims to:

- Reduce reliance on rigid hierarchies.
- Improve organisational responsiveness.
- Enable scalable, data-driven decision-making.

---

## License

This project is for research and prototyping purposes. See `LICENSE` for details.
