import numpy as np

from diff_epi_inference import TimeSeriesDataset, load_timeseries_npz, save_timeseries_npz


def test_save_load_npz_roundtrip(tmp_path):
    ds = TimeSeriesDataset(t=np.array([0.0, 1.0, 2.0]), y=np.array([1, 0, 3]), name="toy")
    path = tmp_path / "toy.npz"

    save_timeseries_npz(ds, path)
    ds2 = load_timeseries_npz(path)

    assert ds2.name == ds.name
    assert np.allclose(ds2.t, ds.t)
    assert np.all(ds2.y == ds.y)
