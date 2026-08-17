from yoyo import step

steps = [
    step(
        '''CREATE TABLE IF NOT EXISTS "skills-evaluator".revisions (
            id                UUID PRIMARY KEY,
            skill_id          TEXT NOT NULL DEFAULT '',
            ref               JSONB,
            skill_name        TEXT NOT NULL DEFAULT '',
            original_skill_md TEXT NOT NULL,
            revised_skill_md  TEXT NOT NULL,
            rationale         TEXT NOT NULL DEFAULT '',
            evaluation        JSONB NOT NULL,
            status            TEXT NOT NULL DEFAULT 'proposed',
            status_reason     TEXT NOT NULL DEFAULT '',
            created_at        TIMESTAMPTZ NOT NULL,
            decided_at        TIMESTAMPTZ
        )''',
        'DROP TABLE IF EXISTS "skills-evaluator".revisions',
    ),
    step(
        '''CREATE INDEX IF NOT EXISTS revisions_skill_idx
           ON "skills-evaluator".revisions (skill_id, created_at DESC)''',
        'DROP INDEX IF EXISTS "skills-evaluator".revisions_skill_idx',
    ),
    step(
        '''CREATE TABLE IF NOT EXISTS "skills-evaluator".evals (
            id         UUID PRIMARY KEY,
            skill_id   TEXT NOT NULL UNIQUE,
            skill_name TEXT NOT NULL DEFAULT '',
            spec       JSONB NOT NULL,
            origin     TEXT NOT NULL DEFAULT 'generated',
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )''',
        'DROP TABLE IF EXISTS "skills-evaluator".evals',
    ),
]
