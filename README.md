# skills-evaluator

A [tapes cassette](https://github.com/papercomputeco/tapes) that evaluates
OpenClaw [Skill Workshop](https://docs.openclaw.ai/tools/skill-workshop)
proposals against real captured session data, built on
[DSPy](https://dspy.ai/).

When OpenClaw evaluates a skill proposal, its `tapes-skills` Gateway plugin
forwards the candidate bundle here. The cassette grounds the judgment in
tapes telemetry through a two-stage DSPy pipeline:

1. **Search** — the skill's name, description, and headings become span
   search queries (`GET /v1/search/spans`) that find the captured sessions
   the skill is actually about; their derived projections render as
   transcripts.
2. **Triage** (`dspy.ChainOfThought(SessionTriage)`) — each transcript is
   classified for the implicit supervision agent traces carry: tool errors,
   retries, loops, corrections, abandonment → `outcome`, `failure_mode`,
   cited `evidence`. The judge reads annotated evidence, not raw logs.
3. **Judgment** (`dspy.ChainOfThought(SkillJudgment)`) — one typed call over
   the proposal, the baseline (for updates), and the annotated evidence
   returns a structured verdict: summary, `pass`/`revise` decision, and
   findings attributed to a fixed rule vocabulary (`accuracy.contradicted`,
   `completeness.missing-step`, `evidence.unsupported`,
   `safety.risky-instruction`, `clarity.ambiguous`).

Two boundaries are deliberate: the cassette never returns `block` (a block
gates apply inside OpenClaw — an LLM judgment over partial evidence advises,
it does not gate), and missing evidence is stated, not papered over (`pass`
in mode `no-evidence` with an `evidence.none` finding).

## Run it

```bash
direnv allow
uv sync

# against a local tapes + ollama (keyless):
CASSETTE_LLM_PROVIDER=ollama CASSETTE_LLM_MODEL=qwen3:0.6b uv run skills-evaluator

# register with tapes:
tapes serve --cassettes=http://127.0.0.1:9978/openapi
```

Tapes republishes the API at `POST /v1/cassettes/skills-evaluator/evaluate`.
Configuration is environment-only, published as schema in
[`src/skills_evaluator/cassette.toml`](./src/skills_evaluator/cassette.toml)
(dots become underscores: `llm.api_key` → `CASSETTE_LLM_API_KEY`). Without an
LLM credential the cassette still starts and answers discovery; `/evaluate`
reports 503.

```bash
curl -s -X POST http://127.0.0.1:8081/v1/cassettes/skills-evaluator/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "proposal": {"id": "prop-1", "kind": "create"},
    "skill": {"name": "morning-catchup", "description": "Daily inbox triage"},
    "candidate": {"skill_md": "# Morning catchup\n\n## Triage inbox\n\n1. ..."}
  }' | jq
```

## Develop

```bash
uv run pytest      # no services or models needed: DSPy modules and tapes are faked
```

The wire contract mirrors OpenClaw's `PluginHookSkillProposalEvaluateResult`
in tapes' snake_case, clamped inside OpenClaw's evaluator caps so a judgment
is never discarded whole for an oversized field. The manifest is authored
once in `cassette.toml` and embedded verbatim into the served OpenAPI
document; request/response schemas are generated from the pydantic wire
models.

## Where DSPy takes this next

The pipeline is deliberately shaped as a DSPy *program* so the optimizer
story from the exploration notes plugs in without restructuring:

- **GEPA over the skill itself** — the triage stage already produces the
  labeled failure evidence a GEPA feedback function needs ("session s3
  failed: the agent called search before checking auth; the skill never
  mentions the precondition"). Evolving a `revise`-decision skill into a
  proposed revision — and feeding it back through OpenClaw's
  `skills.proposals.revise` with `expectedRevisionHash` — is the closed
  loop.
- **GEPA/SIMBA over the pipeline** — `SessionTriage` and `SkillJudgment`
  instructions are themselves optimizable text once a small trainset of
  judged proposals exists; SIMBA's introspective rule mining is a bottom-up
  source of "skills you didn't know you needed."
