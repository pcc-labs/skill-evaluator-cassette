# The behavioral axis: evaluate a stored platform skill by id. The cassette
# resolves the document from tapes and judges it against its provenance
# sessions (plus gated search hits) — mode llm, uncapped score.
#
# Environment-dependent: needs a tapes skill with captured session data.
#
#   hurl \
#     --variables-file vars.env \
#     --variable skill_id=$(curl -s http://127.0.0.1:8081/v1/cassettes/skills | jq -r '.items[0].id') \
#     evidence.hurl

POST {{root_url}}/api/skills-evaluator/evaluate
{
  "ref": {"source": "hurl", "id": "evidence-1"},
  "skill_id": "{{skill_id}}"
}
HTTP 200

[Asserts]
jsonpath "$.mode" == "llm"
jsonpath "$.ref.id" == "evidence-1"
jsonpath "$.metrics.sessions_considered" >= 1
jsonpath "$.score" != null

# The revision history for this skill (may be empty; asserts shape only).
GET {{cassette_url}}/revisions?skill_id={{skill_id}}
HTTP 200

[Asserts]
jsonpath "$.items" exists
