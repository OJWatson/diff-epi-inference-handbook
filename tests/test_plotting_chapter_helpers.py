import numpy as np
import pytest

from diff_epi_inference.plotting import (
    plot_observation_overlay,
    plot_ppc_overlay,
    plot_seir_compartments,
    plot_two_timeseries,
)


def test_plot_seir_compartments_returns_fig_ax():
    t = np.linspace(0.0, 1.0, 5)
    S = np.array([10, 9, 8, 7, 6], dtype=float)
    E = np.array([0, 1, 1, 1, 1], dtype=float)
    infectious = np.array([1, 1, 1, 1, 1], dtype=float)
    R = np.array([0, 0, 1, 2, 3], dtype=float)
    fig, ax = plot_seir_compartments(t=t, S=S, E=E, I_series=infectious, R=R)
    assert fig is ax.figure
    assert len(ax.lines) == 4
    fig.clf()


def test_plot_observation_overlay_requires_matching_shapes():
    t = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match="same shape"):
        plot_observation_overlay(t=t, observed=y, expected=np.arange(4, dtype=float))


def test_plot_two_timeseries_requires_matching_shapes():
    t = np.arange(4, dtype=float)
    y1 = np.arange(4, dtype=float)
    y2 = np.arange(3, dtype=float)
    with pytest.raises(ValueError, match="same shape"):
        plot_two_timeseries(t=t, y1=y1, y2=y2, label1="a", label2="b")


def test_plot_ppc_overlay_draw_shape_validation():
    t = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float)
    bad_draws = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="shape"):
        plot_ppc_overlay(t=t, observed=y, draws=bad_draws)


def test_plot_ppc_overlay_draw_limit():
    t = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float)
    draws = np.tile(y[None, :], (6, 1))
    fig, ax = plot_ppc_overlay(t=t, observed=y, draws=draws, max_draws=3)
    # 3 predictive draws + 1 observed line
    assert len(ax.lines) == 4
    fig.clf()
