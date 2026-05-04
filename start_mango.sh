#!/bin/bash
set -e

enabled_agent_services=()
while IFS= read -r service; do
	if [[ -n "$service" ]]; then
		enabled_agent_services+=("$service")
	fi
done < <(python3 - <<'PY'
import yaml

service_map = {
	"BusinessAgent": "business-agent",
	"CustomerServiceAgent": "customers-agent",
	"HRAgent": "hr-agent",
}

with open("config/agents.yaml", "r", encoding="utf-8") as fh:
	config = yaml.safe_load(fh) or {}

for agent in config.get("agents", []):
	if agent.get("name") == "CentralExecutive":
		continue
	if agent.get("enabled", True):
		service = service_map.get(agent.get("name"))
		if service:
			print(service)
PY
)

core_services=(mango-build-agent mcp ce)
all_services=("${core_services[@]}" "${enabled_agent_services[@]}")

echo "Starting services: ${all_services[*]}"

docker compose down -v
docker compose build mango-build-agent mcp
docker compose run --rm trainer
docker compose up --build "${all_services[@]}"
