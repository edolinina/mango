from agents.worker_network import build_agent_network

MOCK_DIRECTIVES = [
    {
        "task": "Analyze profit margins",
        "capability": "profit_margin",
        "prompt": "Analyze vendor profit margins and provide recommendations.",
        "constraints": "ProfitMargin >= MEAN",
        "context": "Mock context for business strategy analysis.",
        "data_source": None,
        "model": None,
        "validator": {
            "name": "profit_margin",
            "target": "ProfitMargin",
            "features": ["TotalPurchaseDollars", "TotalSalesDollars", "TotalExciseTax", "FreightCost", "GrossProfit"],
            "pass_condition": ">= MEAN"
        },
        "judge_config": {
            "model": "gpt-4o-mini",
            "prompt": "Judge profit margin analysis..."
        },
        "cap": {"name": "profit_margin", "avatar": "💰"},
        "directive": {"task_id": "mock-001"},
        "_id": "mock-001",
        "_agent": "BusinessAgent",
    },
    {
        "task": "Validate gross profit",
        "capability": "gross_profit",
        "prompt": "Check gross profit metrics and report findings.",
        "constraints": "GrossProfit >= PERCENTILE_70",
        "context": "Mock context for business validation.",
        "data_source": None,
        "model": None,
        "validator": {
            "name": "gross_profit",
            "target": "GrossProfit",
            "features": ["TotalPurchaseDollars", "TotalSalesDollars", "TotalExciseTax", "FreightCost"],
            "pass_condition": ">= PERCENTILE_70"
        },
        "judge_config": {
            "model": "gpt-4o-mini",
            "prompt": "Judge gross profit validation..."
        },
        "cap": {"name": "gross_profit", "avatar": "📈"},
        "directive": {"task_id": "mock-002"},
        "_id": "mock-002",
        "_agent": "BusinessAgent",
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
