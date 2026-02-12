import numpy as np

from diff_epi_inference.examples import make_paired_seir_reported_datasets


def test_paired_example_runs_and_time_grids_match():
    ds_det, ds_sto = make_paired_seir_reported_datasets(seed_det=0, seed_sto=1, steps=40)

    assert ds_det.t.shape == ds_det.y.shape == (40,)
    assert ds_sto.t.shape == ds_sto.y.shape == (40,)

    assert np.allclose(ds_det.t, ds_sto.t)

    assert ds_det.name
    assert ds_sto.name

    assert np.all(ds_det.y >= 0)
    assert np.all(ds_sto.y >= 0)
