# mango
docker compose down -v
docker volume rm mango_n8n_data 2>/dev/null || true
docker compose up --build
