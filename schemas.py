from datetime import datetime
from typing import Annotated, Any, Sequence, TypedDict
import uuid

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    # Directive inputs
    agent_name: str
    prompt: str
    task: str
    context: str
    data_path: str
    model: object
    cap: dict
    directive: dict
    _id: str
    validator_features: list
    validator_target: str
    # Graph working state
    analysis_context: str
    validation_feedback: str
    iteration: int
    recommendation_payload: dict
    ml_validation: dict
    # ReAct message history (managed by create_react_agent)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Final output
    results: dict


class Recommendation(BaseModel):
    recommendation: str
    explanation: str
    next_steps: list[str]
    validation_samples: list[dict] = Field(
        default_factory=list,
        description=(
            "5 to 10 sample rows using the validator feature names, representing the target "
            "scenario if the recommendation is followed. Each dict key must be a feature name."
        ),
    )


class MCPEnvelope(BaseModel):
    message_type: str
    sender: str
    target: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: dict[str, Any]
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Directive(BaseModel):
    agent: str
    task: str
    capability: str


class AgentOutput(BaseModel):
    agent: str
    capability: str
    agent_task: str = ""
    validation: str = ""
    results: str
    task_id: str = ""


class MLValidationInput(BaseModel):
    agent_name: str = Field(description="Agent name as configured in agents.yaml")
    capability_name: str = Field(description="Capability name to resolve the validator")
    data_path: str = Field(description="Dataset path for validation")
    validation_samples: list[dict] = Field(
        default_factory=list,
        description="Optional explicit samples to validate instead of loading the dataset",
    )


class DatasetAnalysisInput(BaseModel):
    data_path: str = Field(description="Dataset path")
    feature_cols: list[str] = Field(
        default_factory=list,
        description="Feature columns to include in compact stats and correlations",
    )
    target_col: str = Field(default="", description="Prediction target column for correlations")
