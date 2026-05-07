import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional

import httpx
import yaml
from dotenv import load_dotenv

from schemas import AgentResult, CapabilityResult, TaskResult


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

load_dotenv(ROOT / ".env")

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
DEFAULT_OUT = EVAL_DIR / "results.yaml"


def _capability_name(report_capability: str) -> str:
    if not report_capability:
        return ""
    parts = report_capability.rsplit(" ", 1)
    return parts[0] if len(parts) == 2 else report_capability


def _is_timeout_error_message(message: str) -> bool:
    text = (message or "").lower()
    return "readtimeout" in text or "timed out" in text or "timeout" in text


def _normalize_agent_name(name: str) -> str:
    return (name or "").strip().split(" ")[0]


def _expected_agents_from_directives(directives_text: str) -> set[str]:
    expected: set[str] = set()
    for raw_line in (directives_text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        agent_name = body.split(":", 1)[0].strip()
        if agent_name:
            expected.add(_normalize_agent_name(agent_name))
    return expected


def _directive_task_map_from_directives(directives_text: str) -> dict[str, str]:
    task_map: dict[str, str] = {}
    for raw_line in (directives_text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ":" not in body:
            continue
        agent_part, task_part = body.split(":", 1)
        agent_name = _normalize_agent_name(agent_part.strip())
        task_text = task_part.strip()
        # Trim the capability suffix from UI directives text when present.
        if " (Capability:" in task_text:
            task_text = task_text.split(" (Capability:", 1)[0].strip()
        if agent_name and task_text:
            task_map[agent_name] = task_text
    return task_map


def _load_existing(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return yaml.safe_load(f) or []


def _write(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _parse_feedback(query: str, feedback: dict, directive_task_map: Optional[dict[str, str]] = None) -> TaskResult:
    task_id = ""
    agents: list[AgentResult] = []
    directive_task_map = directive_task_map or {}

    for agent_name, payloads in feedback.items():
        agent_result = AgentResult(agent=agent_name)
        normalized_agent = _normalize_agent_name(agent_name)

        for payload in payloads:
            task_id = payload.get("task_id") or task_id
            report_data = json.loads(payload.get("results") or "{}")
            validation = json.loads(payload.get("validation") or "{}")
            capability_name = _capability_name(payload.get("capability", ""))

            agent_result.capabilities.append(
                CapabilityResult(
                    capability=capability_name,
                    agent_task=(
                        payload.get("agent_task", "").strip()
                        or directive_task_map.get(normalized_agent, "")
                        or query
                    ),
                    iterations=report_data.get("iterations", 0),
                    ml_validation=report_data.get("ml_validation") or validation,
                    recommendation=report_data.get("recommendation", ""),
                    explanation=report_data.get("explanation", ""),
                    next_steps=report_data.get("next_steps", []),
                )
            )

        agents.append(agent_result)

    return TaskResult(
        query=query,
        task_id=task_id,
        agents=agents,
    )


def _trigger_and_collect(
    client: httpx.Client,
    base_url: str,
    task: str,
    poll_interval: float,
) -> TaskResult:
    run_resp = client.post(f"{base_url}/run", json={"task": task}, timeout=None)
    run_resp.raise_for_status()
    run_data = run_resp.json()
    expected_agents: set[str] = set()
    directive_task_map: dict[str, str] = {}

    status = run_data.get("status")
    if status == "error":
        raise RuntimeError(run_data.get("error", "Failed to start run"))
    if status == "running":
        expected_agents = {
            _normalize_agent_name(agent_name)
            for agent_name in (run_data.get("agents") or [])
        }

    if status == "pending":
        directives_text = run_data.get("directives", "")
        expected_agents = _expected_agents_from_directives(directives_text)
        directive_task_map = _directive_task_map_from_directives(directives_text)
        approve_resp = client.post(f"{base_url}/approve", timeout=None)
        approve_resp.raise_for_status()
        approve_data = approve_resp.json()
        if approve_data.get("status") == "error":
            raise RuntimeError(approve_data.get("error", "Failed to approve run"))
        expected_agents = {
            _normalize_agent_name(agent_name)
            for agent_name in (approve_data.get("agents") or [])
        } or expected_agents

    while True:
        time.sleep(poll_interval)
        try:
            results_resp = client.get(f"{base_url}/results", timeout=None)
        except httpx.TimeoutException:
            continue
        results_resp.raise_for_status()
        results_data = results_resp.json()

        if isinstance(results_data, dict) and "status" in results_data:
            if results_data["status"] == "no_results":
                continue
            if results_data["status"] == "error":
                error_message = results_data.get("error", "Workflow error")
                if _is_timeout_error_message(str(error_message)):
                    raise RuntimeError(str(error_message))
                raise RuntimeError(error_message)

        if isinstance(results_data, dict):
            task_result = _parse_feedback(task, results_data, directive_task_map=directive_task_map)
            if expected_agents:
                received_agents = {
                    _normalize_agent_name(agent.agent)
                    for agent in task_result.agents
                    if agent.capabilities
                }
                if not expected_agents.issubset(received_agents):
                    continue
            return task_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tasks in batch through CE APIs and save results locally")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="CE base URL (default from CE_PORT)")
    parser.add_argument("--out", "-o", default=str(DEFAULT_OUT), help="Local output YAML file")
    parser.add_argument("--tasks", nargs="*", help="Override evaluation tasks (space-separated strings)")
    parser.add_argument("--count", "-n", type=int, default=None, help="Run only the first N tasks from the list")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between /results polls")
    parser.add_argument("--include-failed", action="store_true", help="Also persist failed tasks as empty result entries")
    args = parser.parse_args()

    tasks = args.tasks or DEFAULT_TASKS
    if args.count is not None:
        tasks = tasks[: args.count]
    base_url = args.base_url.rstrip("/")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = EVAL_DIR / out_path
    existing = _load_existing(out_path)
    session_results: list[dict] = []
    failed_tasks = 0

    with httpx.Client(timeout=None) as client:
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {task}")
            try:
                task_result = _trigger_and_collect(
                    client=client,
                    base_url=base_url,
                    task=task,
                    poll_interval=args.poll_interval,
                )
                session_results.append(task_result.model_dump(mode="json"))
            except Exception as exc:
                print(f"Task failed: {exc}", file=sys.stderr)
                failed_tasks += 1
                if args.include_failed:
                    task_result = TaskResult(
                        query=task,
                        task_id="",
                        agents=[],
                    )
                    session_results.append(task_result.model_dump(mode="json"))

    existing.extend(session_results)
    _write(out_path, existing)

    successful = sum(1 for r in session_results if r.get("agents"))
    print(f"Done. {successful} successful, {failed_tasks} failed.")
    print(f"Stored locally at {out_path}")


if __name__ == "__main__":
    main()
