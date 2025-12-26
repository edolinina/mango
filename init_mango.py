from pathlib import Path

STRUCTURE = {
    "central_executive": ["ce_agent.py", "mcda.py", "__init__.py"],
    "mcp": ["protocol.py", "message.py", "__init__.py"],
    "agents": {
        "hr": ["agent.py", "models.py", "__init__.py"],
        "infrastructure": ["agent.py", "rl_policy.py", "__init__.py"],
        "business_ops": ["agent.py", "rag.py", "__init__.py"],
        "finance": ["predictor.py", "__init__.py"],
        "__init__.py": None
    },
    "rag": ["retriever.py", "memory.py", "__init__.py"],
    "hitl": ["checkpoints.py", "feedback.py", "__init__.py"],
    "config": ["agents.yaml", "policies.yaml"],
    "evaluation": ["scenarios.py", "metrics.py"],
    "utils": ["logging.py", "registry.py"],
}

def create_tree(base: Path, tree):
    for name, content in tree.items():
        path = base / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_tree(path, content)
        elif isinstance(content, list):
            path.mkdir(parents=True, exist_ok=True)
            for file in content:
                (path / file).touch()
        else:
            path.touch()

if __name__ == "__main__":
    root = Path(".")
    create_tree(root, STRUCTURE)
    print("MANGO project structure created.")

