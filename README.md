# albedo-relight

AI-based single-image albedo estimation and controllable image relighting for
Massey University's 159.333 Programming Project.

The repository is at its initial engineering stage. Model, dataset, training,
and evaluation choices remain open until literature and feasibility review.
The current pipeline is deliberately dependency-light and exists only to prove
that the package and RGB input/output path work end to end.

## Development setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the smoke demo:

```bash
python -m relight
```

Run the checks:

```bash
pytest tests/ -q
ruff check .
```

## Current smoke pipeline

`relight.run_relighting_pipeline` validates a normalized RGB image, treats the
input as placeholder albedo, applies a constant controllable light, and returns
albedo, shading, and relit outputs with matching dimensions. It is not a
scientific albedo or relighting method and will be replaced through later
model-selection Issues.

## Project records

- `docs/decisions.md`: dated technical and scope decisions.
- `docs/diary.md`: human-owned weekly course diary.
- `NOTICE.md`: provenance and licenses for third-party code, models, and data.
