# The full day-zero lifecycle for one skill, end to end:
#
#   evaluate (spec mode) → inspect the generated spec → human-edit it →
#   prove the edit is protected → revise against the edited criteria →
#   accept the revision → prove labels never flip.
#
# Every request names a tapes skill row: the suite runner creates a fresh
# one through POST /v1/skills first (specs are one-per-skill and human
# edits are permanent by design, so a fresh row per run), then passes its
# id in:
#
#   hurl \
#     --variables-file vars.env \
#     --variable skill_id=<uuid from POST /v1/skills> lifecycle.hurl
#
# Or use ./run.sh, which creates the skill and does it for you.

# 1. Day-zero evaluate: the row exists but no captured session relates to
#    it, so any weak search hits are gated, a spec is auto-generated and
#    stored, and the judgment runs in spec mode with the score capped at
#    0.9. The inline candidate is the proposed document for the row.
POST {{cassette_url}}/evaluate
{
  "ref": {"source": "hurl", "id": "lifecycle-1"},
  "skill_id": "{{skill_id}}",
  "candidate": {"skill_md": "Write a Shakespearean sonnet about the given topic."}
}
HTTP 200
[Asserts]
jsonpath "$.mode" == "spec"
jsonpath "$.ref.id" == "lifecycle-1"
jsonpath "$.metrics.sessions_considered" == 0
jsonpath "$.metrics.spec_criteria" >= 1
jsonpath "$.evaluator_version" contains "dspy-rules"

# 2. The generated spec is stored, keyed by the skill id.
GET {{cassette_url}}/evals?skill_id={{skill_id}}
HTTP 200
[Captures]
eval_id: jsonpath "$.id"
generated_sha: jsonpath "$.spec_sha256"
[Asserts]
jsonpath "$.origin" == "generated"
jsonpath "$.skill_id" == "{{skill_id}}"
jsonpath "$.spec.criteria" count >= 1

# 3. The human edit: replace the criteria with our own. Origin flips to
#    edited — from here on, this spec is OUR metric.
PUT {{cassette_url}}/evals/{{eval_id}}
{
  "spec": {
    "criteria": [
      {"id": "inputs-defined", "kind": "structure",
       "description": "Names the inputs it expects (topic, tone...)", "weight": 3},
      {"id": "sonnet-form", "kind": "output-property",
       "description": "Defines 14 lines, iambic pentameter, ABAB CDCD EFEF GG rhyme scheme", "weight": 3},
      {"id": "volta-required", "kind": "output-property",
       "description": "Requires a turn (volta) at line 9 or in the closing couplet", "weight": 2}
    ],
    "cases": [
      {"scenario": "topic=the sea", "expect": ["14 lines", "closing couplet resolves the turn"]}
    ]
  }
}
HTTP 200
[Asserts]
jsonpath "$.origin" == "edited"
jsonpath "$.spec.criteria" count == 3
jsonpath "$.spec_sha256" != "{{generated_sha}}"

# 4. The edit is protected: regeneration refuses, even when forced.
POST {{cassette_url}}/evals?force=true
{
  "skill_id": "{{skill_id}}"
}
HTTP 409

# 5. Revise: the judge checks every one of OUR criteria against the
#    one-line skill, and the reviser rewrites it to satisfy them —
#    grounded in the spec, never evaluator meta-commentary.
POST {{cassette_url}}/revisions
{
  "ref": {"source": "hurl", "id": "lifecycle-1"},
  "skill_id": "{{skill_id}}",
  "candidate": {"skill_md": "Write a Shakespearean sonnet about the given topic."}
}
HTTP 201
[Captures]
revision_id: jsonpath "$.id"
[Asserts]
jsonpath "$.status" == "proposed"
jsonpath "$.skill_id" == "{{skill_id}}"
jsonpath "$.evaluation.mode" == "spec"
jsonpath "$.evaluation.findings" count >= 1
jsonpath "$.revised_skill_md" != ""
jsonpath "$.rationale" != ""

# 6. Label it: the status hook records the human verdict. This row is now
#    one labeled example in the GEPA corpus.
POST {{cassette_url}}/revisions/{{revision_id}}/status
{
  "status": "accepted",
  "reason": "hurl lifecycle: much better than the one-liner"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "accepted"
jsonpath "$.decided_at" != null

# 7. First decision wins: flipping a decided label is refused.
POST {{cassette_url}}/revisions/{{revision_id}}/status
{
  "status": "rejected"
}
HTTP 409

# 8. Re-asserting the same label is idempotent — and the original reason
#    stands (labels don't drift, reasons don't either).
POST {{cassette_url}}/revisions/{{revision_id}}/status
{
  "status": "accepted",
  "reason": "second thoughts, still good"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "accepted"
jsonpath "$.status_reason" == "hurl lifecycle: much better than the one-liner"
