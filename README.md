# MANGO: Managerial Agentic Network for General Orchestration

## Overview

Modern large-scale organizations continue to operate with hierarchical structures and management layers that have remained largely unchanged for decades, despite rapid advances in artificial intelligence. While many technology-driven companies have successfully integrated AI tools to support engineers and knowledge workers—primarily to accelerate coding, documentation, and analysis—these capabilities rarely extend into managerial and executive decision-making layers. As a result, strategic planning, coordination, and organizational control still rely heavily on human-centric processes that struggle to scale, adapt, and incorporate the full breadth of available data.

**MANGO** (Managerial Agentic Network for General Orchestration) is a research framework that explores the possibility of AI systems assuming part of the managerial workload. It investigates agentic AI architectures capable of supporting distributed, context-aware, and collaborative managerial decision-making, with the potential to reduce reliance on rigid organizational hierarchies and improve responsiveness in complex, data-rich enterprise environments.

---


## Research Goals

MANGO is designed to answer the question:  
**Can AI systems be designed to assume part of the managerial workload and enable more adaptive, flatter organizational structures?**

By orchestrating distributed, context-aware, and collaborative AI agents, MANGO aims to:

- Reduce reliance on rigid hierarchies.
- Improve organizational responsiveness.
- Enable scalable, data-driven decision-making.

---

## Key Features

- **Agentic Orchestration:**  
  MANGO coordinates multiple specialized AI agents, each with distinct capabilities and roles, to collaboratively solve complex tasks.

- **Central Executive (CE):**  
  The CE agent receives high-level tasks, generates structured directives, and orchestrates the workflow by delegating subtasks to appropriate agents.

- **Directive Management:**  
  Each task is assigned a unique `directive_id`, ensuring that agent feedback and results are correctly matched to the originating directive, enabling parallel and isolated task execution.

- **Approval Workflow:**  
  MANGO supports both autonomous and human-in-the-loop modes. In non-autonomous mode, the system pauses after directive generation and presents the user with a summary for approval or rejection before proceeding.

- **Agent Feedback Collection:**  
  The system collects and aggregates feedback from agents, filtered by `directive_id`, and presents results in a user-friendly UI.

- **Extensible Agent Design:**  
  Agents can be easily added or configured with different capabilities, roles, and data access, supporting a wide range of enterprise scenarios.

---

## System Architecture

- **Backend:**  
  - Built with FastAPI.
  - Central Executive logic in `central_executive.py`.
  - Asynchronous workflow management and agent communication.
  - REST API endpoints for task submission, approval, rejection, and results polling.

- **Frontend:**  
  - Lightweight JavaScript UI (`static/js/app.js`).
  - Real-time feedback and agent status visualization.
  - Modal dialogs for human approval and result inspection.

- **Agents:**  
  - Modular, capability-driven agents.
  - Communicate via HTTP and a message-passing protocol.
  - Each agent processes directives and returns structured feedback.

---

## Usage

### 1. Start the Service

```bash
./start_mango.sh
```

### 2. Access the UI

Open your browser and navigate to [http://localhost:8000](http://localhost:8001).

### 3. Submit a Task

- Enter a high-level task in the input field and click **Run**.
- In non-autonomous mode, review the generated directives and approve or reject the plan.
- Upon approval, agents are triggered and results are collected and displayed.

---

## API Endpoints

- `POST /run`  
  Submit a new task. Returns agent list and status. In non-autonomous mode, returns status "pending" and directives for approval.

- `POST /approve`  
  Approve the generated directives and start agent execution.

- `POST /reject`  
  Reject the generated directives and cancel the workflow.

- `GET /results?directive_id=...`  
  Poll for agent feedback/results for a specific directive.

- `POST /results/clear?directive_id=...`  
  Clear cached results for a specific directive.

---

## Configuration

- **Autonomous Mode:**  
  Set the environment variable `AUTONOMOUS_MODE=true` to enable fully automated execution without human approval.

- **Agent Configuration:**  
  Agents and their capabilities are defined in the configuration files and can be extended as needed.

---

## Language Model Provider Support

MANGO supports both cloud-based and local language models:

- **OpenAI:**  
  To use OpenAI models, set `LLM_PROVIDER=openai` in your `.env` file and provide your OpenAI API key as the environment variable `OPENAI_API_KEY`.

- **Ollama (Local LLM):**  
  To use a local LLM via [Ollama](https://ollama.com/), set `LLM_PROVIDER=ollama` and configure the `OLLAMA_URL` and `LLM_MODEL` variables in your `.env` file.

You can switch between providers by changing the `LLM_PROVIDER` value in `.env`:

```dotenv
LLM_PROVIDER=ollama   # or 'openai'
```

All relevant configuration options are managed in the `.env` file.

---

## Deployment

MANGO is containerized for easy deployment using Docker Compose.

### Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) must be installed and running on your system.

### Configuration

- All configuration and startup variables are stored in the `.env` file at the project root.  
  Edit this file to set environment variables as needed for your deployment.

### Starting MANGO

From the project root directory, run:

```bash
./start_mango.sh
```

This script will use `docker-compose.yml` to build and start all required services.

### Stopping MANGO

To stop all running containers, use:

```bash
docker-compose down
```

---

## License

This project is for research and prototyping purposes. See `LICENSE` for details.

---

## Acknowledgments

MANGO builds on modern AI, agentic architectures, and orchestration patterns to push the boundaries of AI-assisted management and organizational design.
