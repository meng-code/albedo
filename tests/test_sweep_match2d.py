import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import sweep_match2d as sm  # noqa: E402


def test_parse_config():
    assert sm.parse_config("marco:770:5") == ("marco", 770, 5, "marco_770_5")


def test_parse_config_rejects_other_view_counts():
    with pytest.raises(ValueError):
        sm.parse_config("marco:770:4")
