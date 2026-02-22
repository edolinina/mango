#!/bin/bash
set -e

docker compose down -v
docker compose up --build
docker compose up trainer
docker compose up
