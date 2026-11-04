# Project instructions

This is a Python 3.12 AI/ML project managed with uv.

## Development
- Run Python and tools through uv run.
- Put reusable implementation in src/btp.
- Use notebooks for exploration; keep core logic importable.
- Use type hints for public functions and clear docstrings.
- Follow existing architecture and make focused changes.
- Add dependencies only when needed and explain why.

## Validation
- Run uv run ruff check .
- Run uv run ruff format --check .
- Run uv run pytest for relevant behavioral changes.
- Add tests for bugs, boundaries and invalid inputs.
- Report exactly which checks ran and any failures or skips.

## Experiments
- Use fixed splits and record random seeds.
- Fit preprocessing only on training data.
- Start with a tiny CPU smoke run before expensive training.
- Record config, dataset version and commit for each result.
- Never invent metrics, logs or completed experiments.

## Boundaries
- Do not read or print credentials from .env or auth stores.
- Do not change evaluation criteria merely to improve scores.
- Ask before paid compute, large downloads or long training.
- Leave commits and pushes for my review unless I request them.
