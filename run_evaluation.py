import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

from schemas import EvaluationAgentResult, EvaluationCapabilityResult, EvaluationTaskResult


load_dotenv()

DEFAULT_TASKS = [
    "Increase net income by improving operating efficiency and protecting gross profit.",
    "Reduce customer churn by improving retention drivers such as support, security, and referrals.",
    "Improve workforce performance through training, work-life balance, and manager stability.",
    "Increase profitability while keeping customer churn low and employee performance strong.",
    "Improve customer loyalty without sacrificing business profitability.",
    "Raise employee performance while minimizing customer churn risk.",
    "Improve profitability while keeping both customer loyalty and workforce performance stable.",
    "Increase net income without weakening customer retention or employee performance.",
    "Strengthen customer loyalty for high-tenure customers while maintaining healthy margins.",
    "Achieve balanced improvement in profitability, customer loyalty, and workforce performance.",
]

DEFAULT_BASE_URL = f"http://localhost:{os.getenv('CE_PORT', '8001')}"
DEFAULT_OUT = Path("evaluation/results/results.yaml")


def _capability_name(report_capability: str) -> str:
    if not report_capability:
        return ""
    parts = report_capability.rsplit(" ", 1)
    return parts[0] if len(parts) == 2 else report_capability


def _load_existing(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return yaml.safe_load(f) or []


def _write(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _parse_feedback(query: str, feedback: dict) -> EvaluationTaskResult:
    task_id = ""
    agents: list[EvaluationAgentResult] = []

    for agent_name, payloads in feedback.items():
        agent_result = EvaluationAgentResult(agent=agent_name)

        for payload in payloads:
            task_id = payload.get("task_id") or task_id
            report_data = json.loads(payload.get("results") or "{}")
            validation = json.loads(payload.get("validation") or "{}")
            capability_name = _capability_name(payload.get("capability", ""))

            agent_result.capabilities.append(
                EvaluationCapabilityResult(
                    capability=capability_name,
                    agent_task="",
                    iterations=report_data.get("iterations", 0),
                    ml_validation=report_data.get("ml_validation") or validation,
                    recommendation=report_data.get("recommendation", ""),
                    explanation=report_data.get("explanation", ""),
                    next_steps=report_data.get("next_steps", []),
                )
            )

        agents.append(agent_result)

    return EvaluationTaskResult(
        query=query,
        task_id=task_id,
        created=datetime.now().isoformat(),
        agents=agents,
    )


def _trigger_and_collect(
    client: httpx.Client,
    base_url: str,
    task: str,
    poll_interval: float,
    max_polls: int,
) -> EvaluationTaskResult:
    run_resp = client.post(f"{base_url}/run", json={"task": task})
    run_resp.raise_for_status()
    run_data = run_resp.json()

    status = run_data.get("status")
    if status == "error":
        raise RuntimeError(run_data.get("error", "Failed to start run"))

    if status == "pending":
        approve_resp = client.post(f"{base_url}/approve")
        approve_resp.raise_for_status()
        approve_data = approve_resp.json()
        if approve_data.get("status") == "error":
            raise RuntimeError(approve_data.get("error", "Failed to approve run"))

    for _ in range(max_polls):
        time.sleep(poll_interval)
        results_resp = client.get(f"{base_url}/results")
        results_resp.raise_for_status()
        results_data = results_resp.json()

        if isinstance(results_data, dict) and "status" in results_data:
            if results_data["status"] == "no_results":
                continue
            if results_data["status"] == "error":
                raise RuntimeError(results_data.get("error", "Workflow error"))

        if isinstance(results_data, dict):
            return _parse_feedback(task, results_data)

    raise TimeoutError("Timed out waiting for /results")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation through existing CE APIs and save results locally")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="CE base URL (default from CE_PORT)")
    parser.add_argument("--out", "-o", default=str(DEFAULT_OUT), help="Local output YAML file")
    parser.add_argument("--tasks", nargs="*", help="Override evaluation tasks (space-separated strings)")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between /results polls")
    parser.add_argument("--max-polls", type=int, default=300, help="Max /results polls per task")
    args = parser.parse_args()

    tasks = args.tasks or DEFAULT_TASKS
    base_url = args.base_url.rstrip("/")

    out_path = Path(args.out)
    existing = _load_existing(out_path)
    session_results: list[dict] = []

    with httpx.Client(timeout=args.request_timeout) as client:
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {task}")
            try:
                task_result = _trigger_and_collect(
                    client=client,
                    base_url=base_url,
                    task=task,
                    poll_interval=args.poll_interval,
                    max_polls=args.max_polls,
                )
            except Exception as exc:
                print(f"Task failed: {exc}", file=sys.stderr)
                task_result = EvaluationTaskResult(
                    query=task,
                    task_id="",
                    created=datetime.now().isoformat(),
                    agents=[],
                )

            session_results.append(task_result.model_dump(mode="json"))

    existing.extend(session_results)
    _write(out_path, existing)

    successful = sum(1 for r in session_results if r.get("agents"))
    print(f"Done. {successful}/{len(session_results)} tasks produced agent results.")
    print(f"Stored locally at {out_path}")


if __name__ == "__main__":
    main()
