from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List, Optional
from datetime import datetime
import uuid

class MCPEnvelope(BaseModel):
    message_type: str
    sender: str
    target: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any]
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class Directive(BaseModel):
    agent: str
    task: str
    capability: str

class AgentOutput(BaseModel):
    agent: str
    capability: str
    results: str
    validation: str
    task_id: str
