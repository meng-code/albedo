import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "idarb"))
import sweep_chain_defaults as sc  # noqa: E402
from refine_correspondence import fit_prior  # noqa: E402


def test_standing_value_refers_to_the_standing_run():
    plan = sc.parse_sweep(["prior_sigma=0.1,0.25,affine", "k=64,128", "w_smooth=0.30"])
    assert plan == [
        ("prior_sigma", "0.1", "prior_sigma_0p1"),
        ("prior_sigma", "0.25", "standing"),
        ("prior_sigma", "affine", "prior_sigma_affine"),
        ("k", "64", "standing"),
        ("k", "128", "k_128"),
        ("w_smooth", "0.30", "standing"),
    ]


def test_parse_sweep_rejects_unknown_parameters_and_values():
    with pytest.raises(ValueError):
        sc.parse_sweep(["sigma=0.1"])
    with pytest.raises(ValueError):
        sc.parse_sweep(["k=many"])
    with pytest.raises(ValueError):
        sc.parse_sweep(["k=affine"])


def test_refine_args():
    assert sc.refine_args(None, None) == sc.FIXED_ARGS
    assert sc.refine_args("prior_sigma", "affine").endswith("--rematch_rounds 3 --prior_sigma 0")
    assert sc.refine_args("iters", "4").endswith("--iters 4")


def test_summarise_gives_paired_differences_against_the_standing_run():
    def kp(hits):
        return {f"p{i}": dict(n="5", pck_2=str(h / 5.0), mean="0.03")
                for i, h in enumerate(hits)}

    def tex(w2):
        return {f"p{i}": dict(within_2pct=str(v), within_5pct="0.9", distortion_over_10="0.01",
                              flipped_area="0.1", mutual_anchors="100")
                for i, v in enumerate(w2)}

    plan = [("k", "64", "standing"), ("k", "128", "k_128")]
    out = sc.summarise(dict(standing=kp([3, 3, 3]), k_128=kp([4, 4, 4])),
                       dict(standing=tex([0.6, 0.6, 0.6]), k_128=tex([0.7, 0.7, 0.7])), plan)
    assert out[0]["hits_pck2"] == 9 and out[0]["pck2_diff"] == ""
    assert out[1]["hits_pck2"] == 12 and out[1]["pck2_pooled"] == 80.0
    assert out[1]["pck2_diff"] == 20.0 and out[1]["pck2_ci_low"] == 20.0
    assert out[1]["within_2pct_diff"] == 10.0 and out[1]["mirrored_diff"] == 0.0


def test_fit_prior_without_width_is_the_affine_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    A = rng.normal(size=(3, 3))
    Y = X @ A + 0.5 + 0.05 * rng.normal(size=(40, 3))
    Q = rng.normal(size=(10, 3))
    affine = fit_prior(X, Y, 0.0)(Q)
    assert np.abs(affine - (Q @ A + 0.5)).max() < 0.1
    # with basis functions the anchors themselves are reproduced more closely
    r_aff = np.abs(fit_prior(X, Y, 0.0)(X) - Y).max()
    r_rbf = np.abs(fit_prior(X, Y, 1.0)(X) - Y).max()
    assert r_rbf < r_aff
