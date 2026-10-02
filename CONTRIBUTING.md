<!--
SPDX-FileCopyrightText: 2026 xhdlphzr
SPDX-License-Identifier: MIT
-->

# Contributing to HarmoniaTextor

[![English](https://img.shields.io/badge/English-CONTRIBUTING-007EC6)](https://github.com/xhdlphzr/HarmoniaTextor/blob/main/CONTRIBUTING.md)
[![汉语](https://img.shields.io/badge/汉语-CONTRIBUTING-007EC6)](https://github.com/xhdlphzr/HarmoniaTextor/blob/main/docs/CONTRIBUTING.zh.md)

Thanks for your interest in HarmoniaTextor. This document defines the engineering principles that every contribution must follow.

## Getting started

```console
uv sync --group dev
uv run pytest
uv run ht                     # native pywebview window
uv run flask --app app.app run
```

Requirements: **Python 3.14+** and [uv](https://docs.astral.sh/uv/). The Linux desktop shell needs the `--extra desktop` Qt WebEngine backend.

## Core principles

These are non-negotiable and are enforced by CI.

1. **Deterministic symbolic core.** All musical decisions are implemented as deterministic algorithms; only the LLM is non-deterministic. Checker logic must be provably correct and locked by golden tests.
2. **100% test coverage.** `uv run pytest` is configured to enforce `--cov=src --cov=app --cov-fail-under=100`. Every new bug must first be captured by a failing regression test; review tests for blind spots.
3. **Unify duplicate interfaces.** When two tools or helpers share parameters, factor them into a shared model instead of copying fields.
4. **Lint clean.** `ruff check .` must pass over `src/`, `app/` and `tests/`; prefer `ruff check --fix --unsafe-fixes` before manual edits. Fix issues rather than suppressing them; only fall back to an inline `# noqa` when a violation is genuinely intentional.
5. **Strict typing.** `mypy --strict .` must pass over `src/`, `app/` and `tests/`; run `mypy --install-types` first when stub packages are missing. Prefer fixing annotations; for third-party libraries without stubs, add the matching `types-*` package, and only use an inline `# type: ignore[...]` when no stubs exist and the call cannot be typed.
6. **Google-style docstrings** on every module, class and function, enforced by Ruff's `D` rules with the Google convention (`ruff check --select D .`); the standalone `pydocstyle` tool is not used.
7. **Formatting.** Run `ruff format .` after every change; CI checks `ruff format --check .`.
8. **No dead code.** Remove unreachable or redundant code; keep logic simple.
9. **Feature completeness.** Serve as many realistic workflows as possible within the tool's scope.
10. **Beautiful UI.** Rounded corners, Alex Brush (bundled under `app/static/fonts/`, OFL-1.1) for English display headings, Noto Serif SC with system fallbacks elsewhere; no CDN requests. Slide/expand animations are required.
11. **Reproducible environment.** `uv lock --check` must pass; run `uv lock` and `uv sync` before committing and `uv lock --upgrade` periodically.
12. **REUSE compliance.** `reuse lint` must pass. Every file carries an `SPDX-FileCopyrightText`/`SPDX-License-Identifier` pair, while `LICENSE` and `REUSE.toml` keep `HarmoniaTextor contributors`. Icons under `assets/` are `CC-BY-NC-ND-4.0`.
13. **CI/CD discipline.** CI runs the full gate once on Ubuntu. CD builds PyInstaller bundles on Windows, macOS and Ubuntu and uploads them to the GitHub release as 1 GiB split archives.

## Quality gate (exact order)

```console
uv lock --check
uv run reuse lint
uv run pytest
uv run ruff check .
uv run mypy --strict .
uv run ruff format --check .
uv run ruff check --select D .
```

The final step re-checks Google-style docstrings on modules, classes and functions after formatting. Running `ruff check --fix --unsafe-fixes` locally is encouraged but never committed automatically.

## Development workflow

1. Create a branch.
2. Write the failing test first for any bug or behaviour change.
3. Implement the smallest change that makes the test pass.
4. Run the full quality gate above.
5. Open a pull request describing the change and the test that covers it.

### Test review checklist

- Does every branch have both a positive and a negative case?
- Are empty/`None`/out-of-range inputs covered?
- Do assertions check behaviour rather than implementation details?
- Are state-machine transitions fully exercised?
- Are mocks narrow enough not to hide real bugs?

## Architecture rules

- The symbolic layer (`src/harmoniatextor/techniques`, `checker`, `genres`) must stay free of web and LLM concerns.
- New techniques, check rules and genres are added through their registries — never by editing the engine.
- Parameters are validated with Pydantic models that double as JSON schema for the LangChain tools, the HTTP API and the UI.
- The Flask app and the desktop shell share the same `CompositionService`.

## Licensing

This project is MIT-licensed. By contributing you agree that your contributions are released under the same license. Run `reuse lint` before opening a pull request.
