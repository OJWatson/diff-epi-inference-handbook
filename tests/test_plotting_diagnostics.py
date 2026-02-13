import numpy as np
import pytest

from diff_epi_inference.plotting.diagnostics import ess


def test_ess_uniform_weights_equals_n():
    w = np.ones(10)
    assert ess(w) == pytest.approx(10.0)


def test_ess_degenerate_weights_equals_one():
    w = np.array([1.0, 0.0, 0.0])
    assert ess(w) == pytest.approx(1.0)


def test_ess_rejects_negative():
    with pytest.raises(ValueError):
        ess(np.array([0.2, -0.1, 0.9]))
