# albedo-relight

Single-image albedo estimation and controllable relighting.

## Current state

- **Intrinsic decomposition.** IDArb, vendored at `third_party/IDArb/`, is the working baseline. It runs on
  a GPU and splits an image into albedo, normal, metallic and roughness.
- **Geometry.** `scripts/idarb/normal_to_mesh.py` turns an IDArb normal map into a mesh, using Poisson
  normal integration and screened Poisson surface reconstruction. It runs on a laptop without a GPU.
- **Relighting.** Not implemented yet. `src/relight/` only contains a placeholder pipeline, which treats
  the input image as albedo and multiplies it by a constant light, so that the package and tests exist.

How to run IDArb, the training smoke test and the mesh tool: `scripts/idarb/README.md`.

## Layout

| Path | Contents |
|---|---|
| `src/relight/` | our Python package (placeholder for now) |
| `tests/` | pytest tests |
| `scripts/idarb/` | IDArb helper scripts |
| `third_party/IDArb/` | vendored IDArb code and our changes to it |
| `results/` | small results and sample data |
| `docs/decisions.md` | dated technical decisions |
| `docs/diary.md` | progress log |
| `NOTICE.md` | sources and licenses of third-party code, models and data |

## Package and tests

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest tests/ -q
ruff check .
python -m relight   # runs the placeholder pipeline on a two-pixel image
```
