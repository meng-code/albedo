"""Run a tiny in-memory demonstration of the smoke pipeline."""

from relight import run_relighting_pipeline


def main() -> None:
    """Print the outputs of a deterministic two-pixel smoke run."""

    image = (((0.2, 0.4, 0.6), (0.8, 0.6, 0.4)),)
    result = run_relighting_pipeline(
        image,
        light_intensity=0.5,
        light_color=(1.0, 0.8, 0.6),
    )
    print("albedo:", result.albedo)
    print("shading:", result.shading)
    print("relit:", result.relit)


if __name__ == "__main__":
    main()
