from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from schemas import AgentState

_SYSTEM_TEMPLATE = """{role_instructions}

You are a domain strategist. The dataset has already been analyzed — the results are provided below.

You have one tool:
- validate_recommendation: call this with your generated validation_samples after forming a recommendation.

Workflow:
1. Study the DATA ANALYSIS CONTEXT below and form a clear, action-oriented recommendation.
2. Generate 5-10 validation_samples — one dict per row, using EXACTLY the validator feature names as keys.
3. Call validate_recommendation with those samples.
4. If validation fails, refine your recommendation and repeat from step 2 (max {max_iterations} attempts).
5. When validation succeeds (or retries are exhausted), state your final answer.

VALIDATOR FEATURES (use exactly these as dict keys in validation_samples): {feature_list}
VALIDATOR TARGET (do NOT include in validation_samples): {validator_target}

DATA ANALYSIS CONTEXT:
{analysis_context}

Final answer requirements:
- One clear, concrete action-oriented recommendation in plain English.
- 2-4 sentences of explanation citing the strongest evidence.
- 2-4 short, actionable next steps.
- Do not mention plots, charts, code, Python, dataframe operations, or analysis tooling.
- Focus on the business implication and the action to take."""


def build_system_message(state: AgentState, analysis_context: str = "") -> str:
    features = state.get("validator_features", [])
    return _SYSTEM_TEMPLATE.format(
        role_instructions=state.get("prompt", ""),
        feature_list=", ".join(features) if features else "see dataset",
        validator_target=state.get("validator_target", ""),
        analysis_context=analysis_context or "Not available.",
        max_iterations=3,
    )


def build_human_message(state: AgentState) -> str:
    return (
        f"TASK: {state['task']}\n"
        f"CONTEXT: {state.get('context', '')}"
    )


def build_react_prompt(state: AgentState, analysis_context: str = "") -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=build_system_message(state, analysis_context)),
        MessagesPlaceholder(variable_name="messages"),
    ])
