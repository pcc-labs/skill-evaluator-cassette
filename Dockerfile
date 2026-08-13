# The skills-evaluator cassette image.
#
#   docker build -t tapes/skills-evaluator-cassette:dev .
#
# Two stages: uv resolves and installs the locked environment, then the
# runtime stage carries only the venv and the package source (the project is
# installed editable, and cassette.toml is package data read from source).
# Both stages share the same CPython layout (/usr/local/bin), so the venv's
# interpreter symlinks survive the copy.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS build

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Dependencies first, so editing the cassette does not re-resolve or
# re-download the (DSPy-sized) dependency tree on every rebuild.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY skills_evaluator/ skills_evaluator/
RUN uv sync --frozen --no-dev

FROM python:3.14-slim-bookworm

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/skills_evaluator /app/skills_evaluator
ENV PATH="/app/.venv/bin:$PATH"

# Matches cassette.port in cassette.toml. The operator injects
# CASSETTE_LISTEN with the same port from the TapesCassette spec; this is
# the image's own default for running it bare.
EXPOSE 9978
ENV CASSETTE_LISTEN=0.0.0.0:9978

# DSPy wants a writable disk cache; as a non-root user with no home, point
# it at /tmp so it doesn't fall back to memory-only with a warning.
ENV DSPY_CACHEDIR=/tmp/dspy_cache

USER 65532:65532
ENTRYPOINT ["skills-evaluator"]
