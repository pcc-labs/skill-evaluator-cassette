from urllib.parse import parse_qs, urlsplit

from skills_evaluator.store import SCHEMA, _migration_dsn


def test_migration_dsn_uses_psycopg3_and_cassette_schema():
    dsn = _migration_dsn(
        "postgres://tapes:tapes@postgres:5432/tapes?sslmode=disable"
    )
    parsed = urlsplit(dsn)

    assert parsed.scheme == "postgresql+psycopg"
    assert parse_qs(parsed.query) == {
        "sslmode": ["disable"],
        "schema": [f'"{SCHEMA}"'],
    }
