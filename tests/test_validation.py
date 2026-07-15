"""Parametrized coverage of every input-rejection path in the smoke pipeline.

Each case asserts both the exception type and a message fragment so the
validation contract verified manually in the PR #5 review stays locked in.
"""

import pytest

import relight

VALID_IMAGE = (((0.2, 0.4, 0.6),),)


@pytest.mark.parametrize(
    ("image", "expected_exception", "match"),
    [
        pytest.param((), ValueError, "at least one row", id="empty-image"),
        pytest.param(((),), ValueError, "at least one pixel", id="empty-first-row"),
        pytest.param(
            (((0.0, 0.0, 0.0),), ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))),
            ValueError,
            "rectangular",
            id="ragged-rows",
        ),
        pytest.param(
            (((0.1, 0.2),),),
            ValueError,
            r"image\[0\]\[0\] must contain exactly three RGB channels",
            id="two-channel-pixel",
        ),
        pytest.param(
            (((0.1, 0.2, 0.3, 0.4),),),
            ValueError,
            r"image\[0\]\[0\] must contain exactly three RGB channels",
            id="four-channel-pixel",
        ),
        pytest.param(
            (((True, 0.2, 0.3),),),
            TypeError,
            r"image\[0\]\[0\] channels must be real numbers",
            id="bool-channel",
        ),
        pytest.param(
            ((("0.1", 0.2, 0.3),),),
            TypeError,
            r"image\[0\]\[0\] channels must be real numbers",
            id="string-channel",
        ),
        pytest.param(
            (((-0.1, 0.2, 0.3),),),
            ValueError,
            r"image\[0\]\[0\] channels must be normalized",
            id="negative-channel",
        ),
        pytest.param(
            (((1.1, 0.2, 0.3),),),
            ValueError,
            r"image\[0\]\[0\] channels must be normalized",
            id="channel-above-one",
        ),
    ],
)
def test_rejects_invalid_image(image, expected_exception, match):
    with pytest.raises(expected_exception, match=match):
        relight.run_relighting_pipeline(image)


@pytest.mark.parametrize(
    ("light_color", "expected_exception", "match"),
    [
        pytest.param(
            (1.0, 1.0),
            ValueError,
            "light_color must contain exactly three RGB channels",
            id="two-channels",
        ),
        pytest.param(
            (1.0, 1.0, 1.0, 1.0),
            ValueError,
            "light_color must contain exactly three RGB channels",
            id="four-channels",
        ),
        pytest.param(
            (True, 1.0, 1.0),
            TypeError,
            "light_color channels must be real numbers",
            id="bool-channel",
        ),
        pytest.param(
            ("1.0", 1.0, 1.0),
            TypeError,
            "light_color channels must be real numbers",
            id="string-channel",
        ),
        pytest.param(
            (-0.5, 1.0, 1.0),
            ValueError,
            "light_color channels must be normalized",
            id="negative-channel",
        ),
        pytest.param(
            (1.5, 1.0, 1.0),
            ValueError,
            "light_color channels must be normalized",
            id="channel-above-one",
        ),
    ],
)
def test_rejects_invalid_light_color(light_color, expected_exception, match):
    with pytest.raises(expected_exception, match=match):
        relight.run_relighting_pipeline(VALID_IMAGE, light_color=light_color)


@pytest.mark.parametrize(
    ("light_intensity", "expected_exception", "match"),
    [
        pytest.param(True, TypeError, "light_intensity must be a real number", id="bool"),
        pytest.param("0.5", TypeError, "light_intensity must be a real number", id="string"),
        pytest.param(-0.1, ValueError, "light_intensity must be normalized", id="negative"),
        pytest.param(1.1, ValueError, "light_intensity must be normalized", id="above-one"),
    ],
)
def test_rejects_invalid_light_intensity(light_intensity, expected_exception, match):
    with pytest.raises(expected_exception, match=match):
        relight.run_relighting_pipeline(VALID_IMAGE, light_intensity=light_intensity)
