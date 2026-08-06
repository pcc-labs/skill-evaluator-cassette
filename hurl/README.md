# hurl suite

Executable, asserting version of the end-to-end walkthrough — [Hurl](https://hurl.dev)
is in the dev shell.

| File | What it proves | Needs |
|---|---|---|
| `smoke.hurl` | Discovery anchors: `/ping`, `/openapi` with the embedded manifest | evaluator |
| `lifecycle.hurl` | Day-zero skill: spec-mode judgment → generated spec → human edit → edit protection (409) → spec-grounded revision → accept → label immutability (409) | evaluator + tapes + LLM key |
| `evidence.hurl` | Behavioral axis: evaluate a stored tapes skill by id against its provenance sessions | evaluator + tapes + captured sessions |

## Run everything

```bash
./run.sh
```

It mints a fresh skill name for `lifecycle.hurl` (specs are one-per-skill and
human edits are permanent — reusing a name would trip the 409 the suite
itself asserts) and discovers a `skill_id` for `evidence.hurl` from tapes,
skipping it when none exists.

## Run one file

```bash
hurl --variables-file vars.env --test smoke.hurl
hurl --variables-file vars.env --variable skill=demo-$RANDOM --test lifecycle.hurl
hurl --variables-file vars.env --variable skill_id=<uuid> --test evidence.hurl
```

Drop `--test` to see response bodies instead of a test report.

## Through tapes core

The same suite runs against the proxied route — edit `cassette_url` in
`vars.env` to `http://127.0.0.1:8081/v1/cassettes/skills-evaluator`
(`root_url` stays on the cassette: core probes `/ping` itself, it never
proxies it, so `smoke.hurl` is direct-only). Allow ~30s after evaluator
startup for core's admission refresh.
