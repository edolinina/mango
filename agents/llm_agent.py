from agents.base import BaseAgent
from core.mcp.protocol import AgentOutput


class LLMAgent(BaseAgent):
    async def process_directive(self, directive):
        """Perform agent-specific work on the directive and return an AgentOutput."""
        objective = directive.get("objective") if isinstance(directive, dict) else str(directive)
        result = AgentOutput(
            agent=self.name,
            status="completed",
            summary=f"Processed directive: {objective}",
        )
        return result