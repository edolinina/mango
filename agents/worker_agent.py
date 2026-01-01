import logging
import json

from langchain.agents import create_agent

from mcp_server.protocol import AgentOutput, MCPEnvelope
from agents.worker_network import run_agent_network
from utils.helpers import get_mcp_endpoint

logger = logging.getLogger("mango")


class WorkerAgent:
    def __init__(self, name, config, model, mcp_client, manager_name="CentralExecutive"):
        self.name = name
        self.client = mcp_client
        self.manager = manager_name

        self.tools = config.get("tools", [])
        self.data_sources = config.get("data_sources", [])

        self.model = model

    async def process_agent_directive(self):
        mcp_endpoint = await get_mcp_endpoint(self.client, "list_messages")
        if not mcp_endpoint:
            logger.info("list_messages tool not available on MCP client")
            return None
        
        directive = None
        messages = await mcp_endpoint.ainvoke({})
        for msg in messages:
            parsed = json.loads(msg['text'])

            for entry in parsed:
                if entry.get('message_type') == 'directive' and entry.get('target') == self.name:
                    directive = entry.get('payload')
                    break

        if not directive:
            logger.info(f"No directive found for agent {self.name}")
            return None

        input_state = { 
            "task": directive.get("objective"),
            "tools": self.tools,
            "model": self.model
        }
        result_state = await run_agent_network(input_state)
        summary = json.dumps(result_state.get("results"))

        result = AgentOutput(agent=self.name, status="completed", summary=summary)

        # send feedback back to MCP
        mcp_endpoint = await get_mcp_endpoint(self.client, "send_feedback")
        if mcp_endpoint:
            await mcp_endpoint.ainvoke({
                "envelope": {
                    "message_type": "agent_feedback",
                    "sender": self.name,
                    "target": self.manager,
                    "payload": result.model_dump(),
                }
            })

        return result


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_linear_model(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("lr", LinearRegression()),
        ]
    )

    model.fit(X, y)
    return model
