# Discovery anchors: the two endpoints that make this process a cassette.
# Core probes /ping and fetches /openapi directly; neither is proxied.

GET {{root_url}}/ping
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.cassette" == "skills-evaluator"

# The manifest rides inside the OpenAPI document as a root extension —
# one artifact to fetch, so spec and metadata can never drift apart.
GET {{root_url}}/openapi
HTTP 200
[Asserts]
jsonpath "$['x-tapes-cassette'].kind" == "cassette/v1alpha1"
jsonpath "$['x-tapes-cassette'].cassette.name" == "skills-evaluator"
jsonpath "$['x-tapes-cassette'].tables" count == 2
jsonpath "$.paths['/api/skills-evaluator/evaluate']" exists
jsonpath "$.paths['/api/skills-evaluator/revisions']" exists
jsonpath "$.paths['/api/skills-evaluator/evals']" exists
