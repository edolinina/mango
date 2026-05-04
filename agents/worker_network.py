import asyncio
import json
import logging
import os

from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from agents.dataset_analysis_tool import run_dataset_analysis
from agents.ml_validation_tool import run_ml_validation
from agents.prompts import build_react_prompt, build_human_message
from agents.schemas import AgentState, Recommendation

logger = logging.getLogger("mango")


class ReactAgentEngine:
    MAX_ITERATIONS = 3
    # Each iteration = 1 agent step + 1 tool step; plus 1 final agent step
    _RECURSION_LIMIT = MAX_ITERATIONS * 2 + 1

    async def analyze_node(self, state: AgentState) -> dict:
        data_path = state.get("data_path", "")
        try:
            analysis = run_dataset_analysis.invoke({
                "data_path": data_path,
                "feature_cols": state.get("validator_features", []),
                "target_col": state.get("validator_target", ""),
            })
            analysis_context = json.dumps(analysis, ensure_ascii=True)
        except Exception as exc:
            logger.warning(f"dataset analysis failed: {exc}")
            analysis_context = f"Analysis unavailable: {exc}"

        return {
            "analysis_context": analysis_context,
            "iteration": 0,
            "validation_feedback": "",
        }

    async def react_node(self, state: AgentState) -> dict:
        data_path = state.get("data_path", "")
        model = state["model"]
        agent_name = state.get("agent_name", "")
        cap_name = state.get("cap", {}).get("name", "")

        @tool("validate_recommendation")
        def validate_recommendation(validation_samples: list[dict]) -> dict:
            """Validate recommendation samples with the trained ML validator."""
            return run_ml_validation.invoke({
                "agent_name": agent_name,
                "capability_name": cap_name,
                "data_path": data_path,
                "validation_samples": validation_samples,
            })

        prompt = build_react_prompt(state, state.get("analysis_context", ""))
        react_agent = create_react_agent(
            model,
            [validate_recommendation],
            prompt=prompt,
            response_format=Recommendation,
        )

        human_content = build_human_message(state)
        if state.get("validation_feedback"):
            human_content += (
                f"\n\nVALIDATION FEEDBACK:\n{state.get('validation_feedback')}\n"
                "Refine the recommendation and try validation again."
            )

        result = await react_agent.ainvoke(
            {"messages": [HumanMessage(content=human_content)]},
            config={"recursion_limit": self._RECURSION_LIMIT},
        )

        messages = result.get("messages", [])
        validation_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]

        ml_validation = {}
        if validation_messages:
            try:
                content = validation_messages[-1].content
                ml_validation = json.loads(content) if isinstance(content, str) else content
            except Exception:
                logger.warning("failed to parse validation tool output")

        if not ml_validation:
            ml_validation = {
                "status": "error",
                "error": "Validation tool did not return output",
            }

        structured: Recommendation = result.get("structured_response")
        if structured:
            payload = structured.model_dump(exclude={"validation_samples"})
        else:
            logger.warning("structured_response missing from agent result")
            payload = {"recommendation": "", "explanation": "", "next_steps": []}

        attempt = state.get("iteration", 0) + 1
        feedback = (
            f"Status: {ml_validation.get('status', 'unknown')}. "
            f"Pass rate: {ml_validation.get('pass_rate', 'n/a')}. "
            f"Passed: {ml_validation.get('passed', 0)} / {ml_validation.get('total', 0)}."
        )

        return {
            "iteration": attempt,
            "recommendation_payload": payload,
            "ml_validation": ml_validation,
            "validation_feedback": feedback,
        }

    def _route_after_react(self, state: AgentState) -> str:
        ml_validation = state.get("ml_validation", {})
        attempt = state.get("iteration", 0)
        if ml_validation.get("status") == "success" or attempt >= self.MAX_ITERATIONS:
            return "finalize"
        return "react"

    async def finalize_node(self, state: AgentState) -> dict:
        payload = state.get("recommendation_payload", {})
        ml_validation = state.get("ml_validation", {})
        iteration_count = state.get("iteration", 0)

        logger.info(
            "final recommendation | task=%s | iterations=%s | ml_status=%s | pass_rate=%s",
            state.get("task", ""),
            iteration_count,
            ml_validation.get("status", "unknown"),
            ml_validation.get("pass_rate", "n/a"),
        )

        return {
            "results": {
                **payload,
                "ml_validation": ml_validation,
                "iterations": iteration_count,
            }
        }

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("analyze", self.analyze_node)
        g.add_node("react", self.react_node)
        g.add_node("finalize", self.finalize_node)

        g.add_edge(START, "analyze")
        g.add_edge("analyze", "react")
        g.add_conditional_edges(
            "react",
            self._route_after_react,
            {"react": "react", "finalize": "finalize"},
        )
        g.add_edge("finalize", END)
        return g.compile()

    async def run_directive_branch(self, directive_state: dict) -> dict:
        output = {
            "cap": directive_state.get("cap"),
            "directive": directive_state.get("directive"),
            "_id": directive_state.get("_id"),
        }

        data_path = directive_state.get("data_path", "")
        if not data_path or not os.path.exists(data_path):
            logger.error(f"Invalid data_path '{data_path}' — skipping directive")
            output["results"] = {}
            output["validation"] = {"status": "error", "issues": [f"Dataset not found: {data_path}"]}
            return output

        graph = self._build_graph()
        final_state = await graph.ainvoke(directive_state)
        results = final_state.get("results") or {}

        output["results"] = results
        ml_validation = results.get("ml_validation", {})
        output["validation"] = {
            "status": ml_validation.get("status", "unknown"),
            "issues": [ml_validation.get("error", "")] if ml_validation.get("status") == "error" else [],
        }
        return output

    async def run_agent_reasoning(self, directives: list):
        return list(await asyncio.gather(*(self.run_directive_branch(d) for d in directives)))


async def run_reflection_agent(directives: list):
    return await ReactAgentEngine().run_agent_reasoning(directives)
