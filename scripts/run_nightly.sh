#!/usr/bin/env bash
# Wrapper so launchd can load .env before running the nightly pipeline.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Load secrets — launchd does not inherit shell environment
set -a
# shellcheck source=.env
source "$REPO/.env"
set +a

exec "$REPO/.venv/bin/python" scripts/run_nightly.py
