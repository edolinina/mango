import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from schemas import AgentJudgeResult, CaseJudgeResult, JudgeOutput, TaskResult


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


load_dotenv(ROOT / ".env")

DEFAULT_INPUT = Path(__file__).resolve().parent / "results.yaml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "judged_results.yaml"

SYSTEM_PROMPT = """You are an expert evaluation judge for multi-agent decision recommendations.

Your task is to score each evaluation case on a scale from 1 to 10.

Scoring rubric:
- 1 to 3: poor recommendation quality, weak or irrelevant reasoning, missing or unusable next steps, or failed/contradictory validation evidence.
- 4 to 6: partially useful, but generic, weakly grounded, incomplete, or only somewhat aligned with the user query.
- 7 to 8: good recommendation, aligned with the query, reasonably grounded in the evidence, and operationally useful.
- 9 to 10: excellent recommendation, strongly aligned with the query, specific, evidence-based, coherent, and highly actionable.

Judge each case using these criteria:
- Alignment with the original query
- Relevance of the responding agent and capability
- Specificity and usefulness of the recommendation
- Quality of explanation and whether it is supported by the reported evidence
- Practicality of next steps
- ML validation signal when present

Return only structured output matching the provided schema.
For each CASE, return:
- query
- task_id
- score
- explanation
- agents: one entry per required agent with fields: agent, score, explanation

You MUST include all required agent names exactly as provided by the user prompt in "Required Agents".
If a required agent has weak or missing evidence, still include it with a low score and explanation.
"""


def _resolve_ollama_url(url: str) -> str:
    if "host.docker.internal" in url:
        return url.replace("host.docker.internal", "localhost")
    return url


def _load_judge_model():
    provider = (os.getenv("JUDGE_LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "ollama")).lower()
    model_name = os.getenv("JUDGE_LLM_MODEL") or os.getenv("LLM_MODEL", "gpt-oss:20b")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "ollama":
        base_url = _resolve_ollama_url(os.getenv("OLLAMA_URL", "http://localhost:11434"))
        return ChatOllama(
            base_url=base_url,
            model=model_name,
            temperature=temperature,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")


def _load_results(path: Path) -> list[TaskResult]:
    with open(path) as handle:
        data = yaml.safe_load(handle) or []
    return [TaskResult.model_validate(item) for item in data]


def _dump_case(case: TaskResult) -> str:
    lines = [
        f"Query: {case.query}",
        f"Task ID: {case.task_id}",
    ]

    if not case.agents:
        lines.append("Agents: none")
        return "\n".join(lines)

    for agent in case.agents:
        lines.append(f"Agent: {agent.agent}")
        lines.append(f"Agent Status: {agent.status}")
        if not agent.capabilities:
            lines.append("Capabilities: none")
            continue

        for capability in agent.capabilities:
            lines.extend(
                [
                    f"Capability: {capability.capability}",
                    f"Agent Task: {capability.agent_task}",
                    f"Iterations: {capability.iterations}",
                    f"ML Validation: {capability.ml_validation}",
                    f"Recommendation: {capability.recommendation}",
                    f"Explanation: {capability.explanation}",
                    f"Next Steps: {capability.next_steps}",
                ]
            )

    return "\n".join(lines)


def _required_agent_names(case: TaskResult) -> list[str]:
    return [agent.agent for agent in case.agents]


def _build_case_chain():
    model = _load_judge_model().with_structured_output(CaseJudgeResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Evaluate this case and score it.\n\nRequired Agents: {required_agents}\n\n{case_text}",
            ),
        ]
    )
    return prompt | model


def _normalize_case_judgment(case: TaskResult, judgment: CaseJudgeResult) -> CaseJudgeResult:
    required_agents = _required_agent_names(case)
    if not required_agents:
        return judgment

    seen = {a.agent for a in judgment.agents}
    filled_agents = list(judgment.agents)
    for agent_name in required_agents:
        if agent_name in seen:
            continue
        filled_agents.append(
            AgentJudgeResult(
                agent=agent_name,
                score=judgment.score,
                explanation="Agent-specific judgment was not returned by judge model; case-level score applied.",
            )
        )

    avg_score = int(round(sum(a.score for a in filled_agents) / len(filled_agents))) if filled_agents else judgment.score

    return CaseJudgeResult(
        query=judgment.query or case.query,
        task_id=judgment.task_id or case.task_id,
        score=avg_score,
        explanation=judgment.explanation,
        agents=filled_agents,
    )


def _build_score_chart(judged: JudgeOutput, out_path: Path) -> None:
    rows = []
    for idx, judged_case in enumerate(judged.cases, start=1):
        task_label = f"Task {idx}"
        for agent_judgment in judged_case.agents:
            rows.append({
                "task": task_label,
                "agent": agent_judgment.agent,
                "score": agent_judgment.score,
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    mean_score = df["score"].mean()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    agents = sorted(df["agent"].unique())
    palette = sns.color_palette("tab10", n_colors=len(agents))

    sns.barplot(data=df, x="task", y="score", hue="agent", palette=palette, ax=ax)

    ax.axhline(mean_score, color="crimson", linewidth=1.5, linestyle="--", label=f"Mean ({mean_score:.1f})")

    ax.set_title("Agent Judge Scores by Task")
    ax.set_ylabel("Score (1-10)")
    ax.set_xlabel("")
    ax.set_ylim(0, 10)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Agent", bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge evaluation results with an LLM and output structured scores")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Input YAML results file")
    parser.add_argument("--out", "-o", default=str(DEFAULT_OUTPUT), help="Output YAML for judged cases")
    parser.add_argument("--judge-provider", default=None, help="Override judge LLM provider (ollama or openai), otherwise uses JUDGE_LLM_PROVIDER env var")
    parser.add_argument("--judge-model", default=None, help="Override judge model (otherwise uses JUDGE_LLM_MODEL env var)")
    parser.add_argument("--chart-out", default=None, help="Optional PNG output path for score chart (defaults next to --out)")
    args = parser.parse_args()

    if args.judge_provider:
        os.environ["JUDGE_LLM_PROVIDER"] = args.judge_provider
    if args.judge_model:
        os.environ["JUDGE_LLM_MODEL"] = args.judge_model

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    chart_path = Path(args.chart_out) if args.chart_out else output_path.with_suffix(".png")
    if not chart_path.is_absolute():
        chart_path = (Path.cwd() / chart_path).resolve()

    cases = _load_results(input_path)
    if not cases:
        raise RuntimeError(f"No evaluation cases found in {input_path}")

    case_chain = _build_case_chain()
    judged_cases: list[CaseJudgeResult] = []
    for idx, case in enumerate(cases, start=1):
        print(f"Evaluating agents performance for case {idx}/{len(cases)}: {case.query}")
        case_text = _dump_case(case)
        required_agents = ", ".join(_required_agent_names(case)) or "none"
        raw_judgment = case_chain.invoke({"case_text": case_text, "required_agents": required_agents})
        judged_cases.append(_normalize_case_judgment(case, raw_judgment))

    judged = JudgeOutput(cases=judged_cases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        yaml.dump(judged.model_dump(mode="json"), handle, default_flow_style=False, sort_keys=False)

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    _build_score_chart(judged, chart_path)

    print(f"Judged {len(judged.cases)} case(s).")
    print(f"Stored locally at {output_path}")
    print(f"Chart stored at {chart_path}")


if __name__ == "__main__":
    main()
