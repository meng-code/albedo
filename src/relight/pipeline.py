"""Dependency-light plumbing for the initial relighting pipeline.

The implementation in this module is intentionally a smoke-test placeholder.
It validates the package's data path without claiming to estimate physical
albedo or realistic illumination.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

Pixel = tuple[float, float, float]
Image = tuple[tuple[Pixel, ...], ...]


@dataclass(frozen=True)
class RelightingResult:
    """Outputs produced by the smoke relighting pipeline."""

    albedo: Image
    shading: Image
    relit: Image


def _validate_rgb_triplet(values: Sequence[Real], *, name: str) -> Pixel:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three RGB channels")

    channels: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} channels must be real numbers")
        channel = float(value)
        if not 0.0 <= channel <= 1.0:
            raise ValueError(f"{name} channels must be normalized to [0, 1]")
        channels.append(channel)

    return channels[0], channels[1], channels[2]


def _validate_image(image: Sequence[Sequence[Sequence[Real]]]) -> Image:
    if not image:
        raise ValueError("image must contain at least one row")

    width = len(image[0])
    if width == 0:
        raise ValueError("image rows must contain at least one pixel")

    rows: list[tuple[Pixel, ...]] = []
    for row_index, row in enumerate(image):
        if len(row) != width:
            raise ValueError("image must be rectangular")
        rows.append(
            tuple(
                _validate_rgb_triplet(pixel, name=f"image[{row_index}][{column_index}]")
                for column_index, pixel in enumerate(row)
            )
        )
    return tuple(rows)


def run_relighting_pipeline(
    image: Sequence[Sequence[Sequence[Real]]],
    *,
    light_intensity: Real = 1.0,
    light_color: Sequence[Real] = (1.0, 1.0, 1.0),
) -> RelightingResult:
    """Run the deterministic smoke pipeline on a normalized RGB image.

    The input is used as placeholder albedo. A spatially constant shading map
    is generated from the requested light color and intensity, then multiplied
    with the placeholder albedo to exercise the complete data path.
    """

    albedo = _validate_image(image)
    color = _validate_rgb_triplet(light_color, name="light_color")
    if isinstance(light_intensity, bool) or not isinstance(light_intensity, Real):
        raise TypeError("light_intensity must be a real number")
    intensity = float(light_intensity)
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("light_intensity must be normalized to [0, 1]")

    shading_pixel = tuple(channel * intensity for channel in color)
    shading: Image = tuple(tuple(shading_pixel for _ in row) for row in albedo)
    relit: Image = tuple(
        tuple(
            tuple(
                albedo_channel * shading_channel
                for albedo_channel, shading_channel in zip(pixel, shade, strict=True)
            )
            for pixel, shade in zip(row, shading_row, strict=True)
        )
        for row, shading_row in zip(albedo, shading, strict=True)
    )
    return RelightingResult(albedo=albedo, shading=shading, relit=relit)
