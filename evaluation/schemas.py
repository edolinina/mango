from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CapabilityResult(BaseModel):
    capability: str
    agent_task: str = ""
    iterations: int = 0
    ml_validation: dict[str, Any] = Field(default_factory=dict)
    recommendation: str = ""
    explanation: str = ""
    next_steps: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    agent: str
    status: str = "completed"
    capabilities: list[CapabilityResult] = Field(default_factory=list)


class TaskResult(BaseModel):
    query: str
    task_id: str = ""
    created: str = Field(default_factory=lambda: datetime.now().isoformat())
    agents: list[AgentResult] = Field(default_factory=list)


class AgentJudgeResult(BaseModel):
    agent: str
    score: int = Field(ge=1, le=10)
    explanation: str


class CaseJudgeResult(BaseModel):
    query: str
    task_id: str = ""
    score: int = Field(ge=1, le=10)
    explanation: str
    agents: list[AgentJudgeResult] = Field(default_factory=list)


class JudgeOutput(BaseModel):
    cases: list[CaseJudgeResult] = Field(default_factory=list)
