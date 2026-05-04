# CLI entry point: runs the full CE → agent operational loop directly without a web server.
import json
import asyncio
import traceback

from bootstrap import init_system
from utils.helpers import setup_logger, bold_str

TASK = "Reduce operational costs by 10% without impacting delivery timelines"

logger = setup_logger()


async def operational_loop(task):
    system = init_system()
    ce = system["ce"]
    agents = system["agents"]

    ce_output = await ce.generate_directives(task)
    logger.info(f"Directive for {ce.name}: {task}")
    await ce.send_directives(ce_output)

    logger.info("Waiting for agents to perform directives...")
    all_agent_reports = await asyncio.gather(
        *(agent.process_agent_directive() for agent in agents),
        return_exceptions=True,
    )

    for agent, agent_reports in zip(agents, all_agent_reports):
        if isinstance(agent_reports, Exception):
            logger.error(
                f"Agent {agent.print_name} failed:\n"
                f"{''.join(traceback.format_exception(agent_reports))}"
            )
            continue

        if not agent_reports:
            logger.info(f"Agent {agent.print_name} returned no reports.")
            continue

        logger.info(f"Operational results for agent {agent.print_name}:")
        for report in agent_reports:
            results = json.loads(report.results)
            logger.info(
                f"{bold_str('Capability applied')}: {report.capability}\n"
                f"{bold_str('Recommendation')}: {results['recommendation']}\n"
                f"{bold_str('Explanation')}: {results['explanation']}\n"
                f"{bold_str('Validation')}: {report.validation}\n"
            )
            logger.info("==" * 50)


if __name__ == "__main__":
    asyncio.run(operational_loop(TASK))
