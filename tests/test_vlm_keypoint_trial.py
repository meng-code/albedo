import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import vlm_keypoint_trial as vk  # noqa: E402


def test_assign_views_takes_the_first_view_that_sees_the_keypoint():
    visible = [[True, False, False], [True, True, False]]
    assert vk.assign_views(visible).tolist() == [0, 1, -1]


def test_ruler_to_image_undoes_the_crop_and_the_scale():
    row, col = vk.ruler_to_image((500, 250), (10, 20, 200), 1000)
    assert (row, col) == (10 + 50.0, 20 + 100.0)


def test_nearest_object_vertex_snaps_to_the_object():
    vert = -np.ones((4, 4), dtype=np.int64)
    vert[1, 1], vert[3, 3] = 7, 9
    assert vk.nearest_object_vertex((0.2, 0.1), vert) == 7
    assert vk.nearest_object_vertex((3.9, 3.0), vert) == 9
