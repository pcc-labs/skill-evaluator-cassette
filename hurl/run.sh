#!/usr/bin/env bash
# Runs the hurl suite against a live evaluator.
#
# Prerequisites (each in its own terminal):
#   tapes serve --cassettes=http://127.0.0.1:9978/openapi
#   TAPES_DATABASE_URL=... CASSETTE_LLM_PROVIDER=openai uv run skills-evaluator
#
# lifecycle.hurl needs a unique skill name per run (specs are one-per-skill
# and human edits are permanent by design) — this script mints one.
#
# evidence.hurl runs only when a tapes skill id is available (pass SKILL_ID
# or let the script ask tapes for the first stored skill).
set -euo pipefail
cd "$(dirname "$0")"

TAPES_API="${TAPES_API:-http://127.0.0.1:8081}"
SKILL="demo-$(date +%s)-$RANDOM"

echo "== smoke =="
hurl --variables-file vars.env --test smoke.hurl

echo "== lifecycle (skill: $SKILL) =="
hurl --variables-file vars.env --variable "skill=$SKILL" --test lifecycle.hurl

SKILL_ID="${SKILL_ID:-$(curl -sf "$TAPES_API/v1/skills" 2>/dev/null | jq -r '.items[0].id // empty' || true)}"
if [ -n "$SKILL_ID" ]; then
  echo "== evidence (skill_id: $SKILL_ID) =="
  hurl --variables-file vars.env --variable "skill_id=$SKILL_ID" --test evidence.hurl
else
  echo "== evidence: skipped (no stored tapes skill found at $TAPES_API; set SKILL_ID to run) =="
fi
