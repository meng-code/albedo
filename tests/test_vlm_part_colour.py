import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import vlm_part_colour as vp  # noqa: E402


def test_best_views_prefers_the_visible_view_that_faces_the_point():
    visible = [[True, True, False], [True, False, False]]
    facing = [[0.2, 0.9, 1.0], [0.8, 1.0, 1.0]]
    assert vp.best_views(visible, facing).tolist() == [1, 0, -1]


def test_disc_colour_is_the_median_of_object_pixels_and_falls_back_to_the_nearest():
    photo = np.zeros((6, 6, 3), dtype=np.uint8)
    photo[:, :3] = 10
    photo[:, 3:] = 200
    mask = np.zeros((6, 6), dtype=bool)
    mask[:, :2] = True
    assert vp.disc_colour(photo, mask, (2.5, 1.0), 1.5).tolist() == [10, 10, 10]
    # far from the object: the nearest object pixel
    assert vp.disc_colour(photo, mask, (2.5, 5.5), 0.5).tolist() == [10, 10, 10]


def test_pixel_colours_take_the_nearest_object_pixel():
    photo = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = mask[3, 3] = True
    out = vp.pixel_colours(photo, mask, np.array([[0.0, 0.0], [3.9, 3.9]]))
    assert out.tolist() == [photo[1, 1].tolist(), photo[3, 3].tolist()]
