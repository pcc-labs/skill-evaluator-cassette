from yoyo import step

steps = [
    step(
        """ALTER TABLE "skills-evaluator".evals
           ADD COLUMN IF NOT EXISTS source_sha256 TEXT NOT NULL DEFAULT ''""",
        '''ALTER TABLE "skills-evaluator".evals
           DROP COLUMN IF EXISTS source_sha256''',
    ),
]
