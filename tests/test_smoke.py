import pytest

import relight


def test_smoke_pipeline_preserves_shape_and_applies_light() -> None:
    image = (
        ((0.2, 0.4, 0.6), (0.8, 0.6, 0.4)),
        ((1.0, 0.5, 0.0), (0.1, 0.2, 0.3)),
    )

    result = relight.run_relighting_pipeline(
        image,
        light_intensity=0.5,
        light_color=(1.0, 0.8, 0.6),
    )

    assert result.albedo == image
    assert len(result.shading) == len(image)
    assert all(len(row) == len(image[0]) for row in result.shading)
    assert result.shading[0][0] == pytest.approx((0.5, 0.4, 0.3))
    assert result.relit[0][0] == pytest.approx((0.1, 0.16, 0.18))


def test_smoke_pipeline_rejects_ragged_images() -> None:
    image = (((0.0, 0.0, 0.0),), ((1.0, 1.0, 1.0), (0.5, 0.5, 0.5)))

    with pytest.raises(ValueError, match="rectangular"):
        relight.run_relighting_pipeline(image)


def test_package_version() -> None:
    assert relight.__version__ == "0.1.0"
