from agents.worker_network import build_agent_network

MOCK_DIRECTIVES = [
    {
        "task": "Analyze cost efficiency",
        "capability": "cost_analysis",
        "prompt": "Analyze the data and provide recommendations.",
        "constraints": "infra_cost <= MEAN",
        "context": "Mock context for visualization.",
        "data_source": None,
        "model": None,
        "validator": {
            "name": "cost_validator",
            "target": "infra_cost",
            "features": ["traffic", "capacity"],
            "pass_condition": "<= MEAN"
        },
        "judge_config": {
            "model": "gpt-5-mini",
            "prompt": "Judge prompt..."
        },
        "cap": {"name": "cost_analysis", "avatar": "📊"},
        "directive": {"task_id": "mock-001"},
        "_id": "mock-001",
        "_agent": "ITAgent",
    },
    {
        "task": "Validate SLA compliance",
        "capability": "sla_check",
        "prompt": "Check SLA metrics and report.",
        "constraints": "sla_met >= 1",
        "context": "Mock context for visualization.",
        "data_source": None,
        "model": None,
        "validator": {
            "name": "sla_validator",
            "target": "sla_met",
            "features": ["traffic", "capacity"],
            "pass_condition": ">= 1"
        },
        "judge_config": {
            "model": "gpt-5-mini",
            "prompt": "Judge prompt..."
        },
        "cap": {"name": "sla_check", "avatar": "✅"},
        "directive": {"task_id": "mock-002"},
        "_id": "mock-002",
        "_agent": "ITAgent",
    },
]


def visualize_agent_network(directives: list = None):
    """Visualize the agent network graph. Saves as PNG if possible, else prints Mermaid."""
    graph = build_agent_network(directives or MOCK_DIRECTIVES)
    app = graph.compile()
    try:
        png = app.get_graph().draw_mermaid_png()
        path = "agent_network.png"
        with open(path, "wb") as f:
            f.write(png)
        print(f"Graph saved to {path}")
    except Exception:
        print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    visualize_agent_network()
