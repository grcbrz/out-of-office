#!/usr/bin/env bash
# Wrapper so launchd can load .env before running the nightly pipeline.
set -euo pipefail

REPO="/Users/gracebraz/code/grcbrz/out-of-office"
cd "$REPO"

# Load secrets — launchd does not inherit shell environment
set -a
# shellcheck source=.env
source "$REPO/.env"
set +a

exec "$REPO/.venv/bin/python" scripts/run_nightly.py
