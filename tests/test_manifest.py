"""The drift gate between the authored manifest and the served one.

The repo-root ``cassette.toml`` is what a deployment reads; the server's
in-code ``MANIFEST`` is what core discovers under ``x-tapes-cassette``. The
server never reads the TOML, so the only thing holding them equal is this
test: it parses the very file a deployer would and asserts it derives
exactly what the server serves.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from skills_evaluator.server import MANIFEST

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "cassette.toml"


def test_authored_manifest_matches_served():
    authored = tomllib.loads(MANIFEST_PATH.read_text())
    assert authored == MANIFEST


def test_manifest_identity_is_consistent():
    # The identity everything derives from: route, schema, role, and the
    # port the Dockerfile exposes all follow the name and port here.
    assert MANIFEST["cassette"]["name"] == "skills-evaluator"
    assert MANIFEST["cassette"]["port"] == 9978
    assert MANIFEST["api"]["prefix_path"] == "api"
