# 🍳 skills-evaluator-cassette

An agnostic [tapes cassette](https://github.com/papercomputeco/tapes) that evaluates provided
skill documents against real captured session data, built on
[DSPy GEPA optimizers](https://dspy.ai/getting-started/gepa-optimization/).

---

## 🧪 How it works

`POST /evaluate` takes a skill (inline
markdown, or a tapes `skill_id` the cassette resolves itself), an optional
baseline it would replace, and opaque `ref` correlation metadata echoed back
in the response.

Hosts conform through thin adapters — OpenClaw's
[Skill Workshop](https://docs.openclaw.ai/tools/skill-workshop) plugs in via
the `openclaw-tapes` Openclaw Gateway plugin. Paper platform callers (console, CLI, a
generation-time quality gate) can POST a `skill_id` directly.

The cassette grounds the judgment in tapes telemetry through a
two-stage DSPy pipeline:

1. **Evidence selection** — a skill resolved by `skill_id` brings its
   **provenance** (`originatingSessionIds`, the sessions it was generated
   from), which outranks anything search can find; span search
   (`GET /v1/search/spans`) over the skill's name, description, and headings
   tops up the evidence with newer related sessions. Chosen sessions render
   as transcripts.
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

## 🏃 Run it

```bash
# Enter the nix shell
direnv allow

# Make sure you have a uv venv setup!
uv sync

# against a local tapes + ollama (keyless):
CASSETTE_LLM_PROVIDER=ollama CASSETTE_LLM_MODEL=qwen3:0.6b uv run skills-evaluator

# register with tapes:
tapes serve --cassettes=http://127.0.0.1:9978/openapi
```

Tapes republishes the API at `POST /v1/cassettes/skills-evaluator/evaluate`.
Configuration is environment-only, published as schema in
[`skills_evaluator/cassette.toml`](./skills_evaluator/cassette.toml)
(dots become underscores: `llm.api_key` → `CASSETTE_LLM_API_KEY`). Without an
LLM credential the cassette still starts and answers discovery; `/evaluate`
reports 503.

```bash
# Inline document (what the OpenClaw plugin sends):
curl -s -X POST http://127.0.0.1:8081/v1/cassettes/skills-evaluator/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "ref": {"source": "openclaw", "id": "prop-1"},
    "skill": {"name": "morning-catchup", "description": "Daily inbox triage"},
    "candidate": {"skill_md": "# Morning catchup\n\n## Triage inbox\n\n1. ..."}
  }' | jq

# A platform skill by id — resolved from tapes, judged against its
# provenance sessions first:
curl -s -X POST http://127.0.0.1:8081/v1/cassettes/skills-evaluator/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"skill_id": "<uuid from GET /v1/skills>"}' | jq '{decision, score, metrics}'
```

The response carries `decision` (`pass`/`revise`), attributed `findings`, a
0..1 `score` (null when there was no evidence — a GEPA-ready metric), and
evidence metrics including `provenance_sessions`. Passing `baseline` frames
the evaluation as an update replacing it; there is no separate "kind" field.

## 🔁 Revisions: closing the loop

Evaluation says *what's wrong*; revisions propose *the fix* and record what
humans thought of it. `POST /revisions` (same body as `/evaluate`) runs the
evaluation pipeline, then a `SkillRevisionProposal` DSPy step rewrites the
skill so the findings are addressed — grounded only in the session evidence
(409 when there is none: no ungrounded rewrites).

Each revision is stored `proposed` in the cassette's **own Postgres table**
(`"skills-evaluator".revisions`, declared in the manifest, migrated at
startup when `TAPES_DATABASE_URL` is set; memory-only otherwise) with a
loose `skill_id` link back to the tapes skills table, so a skill's whole
revision history hangs together.

Hosts report the human verdict through the status hook:

```bash
# propose a revision for a platform skill
curl -s -X POST .../v1/cassettes/skills-evaluator/revisions \
  -d '{"skill_id": "<uuid>"}' | jq '{id, status, revised_skill_md}'

# the labeling hook: accepted | rejected (decided labels never flip — 409)
curl -s -X POST .../v1/cassettes/skills-evaluator/revisions/<id>/status \
  -d '{"status": "accepted", "reason": "applied in the console"}'

# a skill's revision history
curl -s ".../v1/cassettes/skills-evaluator/revisions?skill_id=<uuid>"
```

Every decided row is one labeled example — (skill + evidence, proposed
rewrite, human verdict) — which is precisely the trainset a GEPA run needs.
The corpus builds itself as a side effect of normal review.

## 🌲 Develop

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

- **GEPA over the skill itself** — the pieces now exist end to end: the
  triage stage produces the labeled failure evidence a feedback function
  needs, `/revisions` produces candidate rewrites, and the status hook
  accumulates accepted/rejected labels in the revisions table. A GEPA run
  is "generate candidates with `SkillReviser`, score them with `/evaluate`,
  seed the reflection LM with the decided corpus" — then feed winners back
  through OpenClaw's `skills.proposals.revise` or a new tapes skill
  version.
- **GEPA/SIMBA over the pipeline** — `SessionTriage` and `SkillJudgment`
  instructions are themselves optimizable text once a small trainset of
  judged proposals exists; SIMBA's introspective rule mining is a bottom-up
  source of "skills you didn't know you needed."
