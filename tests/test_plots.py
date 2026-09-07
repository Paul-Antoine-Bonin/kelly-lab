"""Tests for the wealth path chart.

Matplotlib is forced onto the Agg backend so nothing tries to open a window.
The checks are about the data that reaches the canvas, not about pixels: that
every fraction is replayed on the same coin flips, that ruined paths are
floored rather than dropped, and that the axes are labelled.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from kelly.plots import RUIN_FLOOR, plot_wealth_paths  # noqa: E402
from kelly.simulate import (  # noqa: E402
    BetSpec,
    draw_wins,
    log_paths_from_wins,
    simulate_log_paths,
)

EDGE = BetSpec(p=0.5, b=2.0)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# --- the shared draw ---------------------------------------------------------


def test_draw_wins_reproduces_the_old_path_draw():
    # The refactor must not move the figures: the same seed has to give the
    # same paths as before wins were split out.
    wins = draw_wins(EDGE, horizon=50, n_paths=4, seed=7)
    assert np.allclose(
        log_paths_from_wins(wins, EDGE, 0.25),
        simulate_log_paths(EDGE, 0.25, horizon=50, n_paths=4, seed=7),
    )


def test_the_same_wins_give_the_same_win_and_loss_positions():
    wins = draw_wins(EDGE, horizon=100, n_paths=1, seed=3)
    small = np.diff(log_paths_from_wins(wins, EDGE, 0.1)[0]) > 0
    large = np.diff(log_paths_from_wins(wins, EDGE, 0.7)[0]) > 0
    # Different stake, identical sequence of wins and losses.
    assert np.array_equal(small, large)
    assert np.array_equal(small, wins[0])


def test_a_single_row_of_wins_is_accepted():
    paths = log_paths_from_wins(np.array([True, False, True]), EDGE, 0.5)
    assert paths.shape == (1, 4)


def test_log_paths_from_wins_refuses_a_fraction_above_one():
    with pytest.raises(ValueError):
        log_paths_from_wins(draw_wins(EDGE, 10, seed=0), EDGE, 1.5)


# --- what reaches the canvas -------------------------------------------------


def test_one_line_per_fraction_per_path():
    ax = plot_wealth_paths(EDGE, [0.1, 0.25, 0.5], horizon=60, n_paths=3, seed=0)
    assert len(ax.lines) == 3 * 3 + 1  # plus the bankroll reference line


def test_every_fraction_is_labelled_once():
    ax = plot_wealth_paths(EDGE, [0.1, 0.5], horizon=40, n_paths=4, seed=0)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["f = 0.10", "f = 0.50"]


def test_the_plotted_wealth_matches_the_simulator():
    ax = plot_wealth_paths(EDGE, [0.25], horizon=80, n_paths=1, seed=11)
    drawn = ax.lines[0].get_ydata()
    wins = draw_wins(EDGE, horizon=80, n_paths=1, seed=11)
    expected = np.exp(log_paths_from_wins(wins, EDGE, 0.25))[0]
    assert np.allclose(drawn, expected)


def test_the_axis_is_logarithmic_and_starts_at_the_bankroll():
    ax = plot_wealth_paths(EDGE, [0.25], horizon=30, seed=0)
    assert ax.get_yscale() == "log"
    assert ax.lines[0].get_ydata()[0] == pytest.approx(1.0)


def test_ruin_is_floored_rather_than_dropped():
    # Staking everything: one loss ends the run at exactly zero, which a log
    # axis cannot draw. It has to appear at the floor, not disappear.
    ax = plot_wealth_paths(EDGE, [1.0], horizon=40, n_paths=1, seed=1)
    drawn = ax.lines[0].get_ydata()
    assert np.all(np.isfinite(drawn))
    assert drawn.min() == pytest.approx(RUIN_FLOOR)
    assert drawn[-1] == pytest.approx(RUIN_FLOOR)


def test_the_floor_never_moves_a_surviving_path():
    ax = plot_wealth_paths(EDGE, [0.25], horizon=50, n_paths=1, seed=5)
    assert ax.lines[0].get_ydata().min() > RUIN_FLOOR


def test_the_axes_are_labelled():
    ax = plot_wealth_paths(EDGE, [0.25], horizon=20, seed=0)
    assert "Bet" in ax.get_xlabel()
    assert "Wealth" in ax.get_ylabel()
    assert ax.get_title()


def test_an_existing_axes_is_used_when_given():
    _, ax = plt.subplots()
    assert plot_wealth_paths(EDGE, [0.25], horizon=20, seed=0, ax=ax) is ax


# --- validation --------------------------------------------------------------


def test_at_least_one_fraction_is_required():
    with pytest.raises(ValueError):
        plot_wealth_paths(EDGE, [], horizon=20)


def test_the_horizon_must_have_at_least_one_bet():
    with pytest.raises(ValueError):
        plot_wealth_paths(EDGE, [0.25], horizon=0)


def test_the_floor_must_be_positive():
    with pytest.raises(ValueError):
        plot_wealth_paths(EDGE, [0.25], horizon=20, floor=0.0)
