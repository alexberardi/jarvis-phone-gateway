#!/bin/bash
# Dev runner — Docker on every platform (no GPU dependency; see compose note).
set -e
cd "$(dirname "$0")"
docker compose -f docker-compose.dev.yaml up -d --build
docker compose -f docker-compose.dev.yaml logs -f jarvis-phone-gateway
