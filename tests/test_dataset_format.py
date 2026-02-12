import numpy as np
import pytest

from diff_epi_inference import TimeSeriesDataset, from_cases


def test_from_cases_constructs_regular_time_grid():
    y = np.array([0, 1, 2])
    ds = from_cases(y=y, dt=0.5, t0=1.0, name="cases")

    assert ds.name == "cases"
    assert np.allclose(ds.t, np.array([1.0, 1.5, 2.0]))
    assert np.all(ds.y == y)


def test_dataset_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        TimeSeriesDataset(t=np.array([0.0, 0.0]), y=np.array([1, 2]))  # not strictly increasing

    with pytest.raises(ValueError):
        TimeSeriesDataset(t=np.array([0.0, 1.0]), y=np.array([-1, 2]))  # negative

    with pytest.raises(ValueError):
        TimeSeriesDataset(t=np.array([0.0, 1.0]), y=np.array([1]))  # length mismatch
