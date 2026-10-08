"""Tests for locating the optimal stake.

The closed form and the search are tested separately first, then against each
other. The simulated check uses a fixed seed and a tolerance set from the
sampling error of the win rate, so it cannot flake but would catch a peak in
the wrong place.
"""

import math

import numpy as np
import pytest

from kelly.optimal import (
    golden_section_max,
    growth_rate,
    kelly_fraction,
    numerical_kelly_fraction,
    simulated_growth_curve,
)
from kelly.simulate import BetSpec, final_log_wealth

EDGE = BetSpec(p=0.5, b=2.0)

# A spread of bets with an interior optimum: short and long odds, small and
# large edges.
BETS = [
    BetSpec(p=0.5, b=2.0),
    BetSpec(p=0.6, b=1.0),
    BetSpec(p=0.55, b=1.0),
    BetSpec(p=0.3, b=4.0),
    BetSpec(p=0.1, b=12.0),
    BetSpec(p=0.9, b=0.2),
]


# --- the growth rate ---------------------------------------------------------


def test_growth_rate_is_zero_when_nothing_is_staked():
    assert growth_rate(EDGE, 0.0) == 0.0


def test_growth_rate_matches_the_formula():
    f = 0.25
    expected = 0.5 * math.log(1 + 0.5) + 0.5 * math.log(1 - 0.25)
    assert growth_rate(EDGE, f) == pytest.approx(expected)


def test_growth_rate_accepts_an_array():
    fs = np.array([0.0, 0.25, 0.5])
    g = growth_rate(EDGE, fs)
    assert g.shape == (3,)
    assert g[1] == pytest.approx(growth_rate(EDGE, 0.25))


def test_staking_everything_on_a_bet_that_can_lose_is_minus_infinity():
    assert growth_rate(EDGE, 1.0) == -math.inf


def test_staking_everything_on_a_certain_win_is_finite():
    # q * log(0) would be 0 * -inf = nan if computed naively.
    certain = BetSpec(p=1.0, b=2.0)
    assert growth_rate(certain, 1.0) == pytest.approx(math.log(3.0))


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_growth_rate_refuses_fractions_outside_zero_to_one(fraction):
    with pytest.raises(ValueError):
        growth_rate(EDGE, fraction)


def test_growth_rate_matches_the_simulator():
    # Same quantity, two routes: the closed form here and the mean of the
    # simulator's final log wealth.
    sim = final_log_wealth(EDGE, 0.25, horizon=500, n_paths=50_000, seed=0)
    assert np.mean(sim) / 500 == pytest.approx(growth_rate(EDGE, 0.25), abs=2e-4)


# --- the closed form ---------------------------------------------------------


def test_kelly_fraction_for_the_coin_paying_two_to_one():
    # 0.5 - 0.5 / 2
    assert kelly_fraction(EDGE) == pytest.approx(0.25)


@pytest.mark.parametrize("spec", BETS)
def test_kelly_fraction_is_edge_over_odds(spec):
    assert kelly_fraction(spec) == pytest.approx(spec.edge / spec.b)


@pytest.mark.parametrize("spec", BETS)
def test_the_derivative_vanishes_at_the_kelly_fraction(spec):
    f = kelly_fraction(spec)
    h = 1e-6
    slope = (growth_rate(spec, f + h) - growth_rate(spec, f - h)) / (2 * h)
    assert slope == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("spec", [BetSpec(p=0.5, b=1.0), BetSpec(p=0.4, b=1.0)])
def test_without_an_edge_the_best_stake_is_nothing(spec):
    assert kelly_fraction(spec) == 0.0


def test_a_certain_win_is_staked_in_full():
    assert kelly_fraction(BetSpec(p=1.0, b=0.5)) == 1.0


# --- the search --------------------------------------------------------------


def test_golden_section_finds_the_peak_of_a_parabola():
    assert golden_section_max(lambda x: -(x - 0.3) ** 2) == pytest.approx(0.3, abs=1e-6)


def test_golden_section_finds_a_peak_on_the_lower_boundary():
    assert golden_section_max(lambda x: -x) == 0.0


def test_golden_section_finds_a_peak_on_the_upper_boundary():
    assert golden_section_max(lambda x: x) == 1.0


def test_golden_section_copes_with_minus_infinity_at_an_end():
    func = lambda x: -math.inf if x == 1.0 else -(x - 0.7) ** 2  # noqa: E731
    assert golden_section_max(func) == pytest.approx(0.7, abs=1e-6)


def test_golden_section_refuses_an_empty_bracket():
    with pytest.raises(ValueError):
        golden_section_max(lambda x: x, lo=1.0, hi=0.0)


@pytest.mark.parametrize("spec", BETS)
def test_searching_the_exact_growth_rate_recovers_the_formula(spec):
    # No simulation noise here: this isolates the search itself.
    found = golden_section_max(lambda f: growth_rate(spec, f))
    assert found == pytest.approx(kelly_fraction(spec), abs=1e-6)


# --- the simulated curve -----------------------------------------------------


def test_the_simulated_curve_is_zero_when_nothing_is_staked():
    curve = simulated_growth_curve(EDGE, horizon=100, n_paths=1_000, seed=0)
    assert curve(0.0) == 0.0


def test_the_simulated_curve_reuses_one_draw():
    # Asking twice about the same fraction must give exactly the same answer,
    # otherwise the search would be chasing noise.
    curve = simulated_growth_curve(EDGE, horizon=100, n_paths=1_000, seed=0)
    assert curve(0.3) == curve(0.3)


def test_the_simulated_curve_is_minus_infinity_at_full_stake():
    curve = simulated_growth_curve(EDGE, horizon=100, n_paths=1_000, seed=0)
    assert curve(1.0) == -math.inf


def test_the_simulated_curve_at_full_stake_on_a_certain_win():
    curve = simulated_growth_curve(BetSpec(p=1.0, b=2.0), horizon=10, n_paths=5, seed=0)
    assert curve(1.0) == pytest.approx(math.log(3.0))


def test_the_simulated_curve_matches_the_simulator():
    curve = simulated_growth_curve(EDGE, horizon=500, n_paths=10_000, seed=3)
    sim = final_log_wealth(EDGE, 0.4, horizon=500, n_paths=10_000, seed=3)
    # Same seed, same binomial draw, so this is equality up to rounding.
    assert curve(0.4) == pytest.approx(np.mean(sim) / 500, rel=1e-12)


def test_the_simulated_curve_leaves_the_global_state_alone():
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    simulated_growth_curve(EDGE, horizon=100, n_paths=100, seed=0)
    assert np.random.random() == before


@pytest.mark.parametrize("bad", [{"horizon": 0}, {"n_paths": 0}])
def test_the_simulated_curve_refuses_empty_simulations(bad):
    kwargs = {"horizon": 10, "n_paths": 10, **bad}
    with pytest.raises(ValueError):
        simulated_growth_curve(EDGE, seed=0, **kwargs)


# --- simulation against the formula ------------------------------------------


@pytest.mark.parametrize("spec", BETS)
def test_the_simulated_optimum_matches_the_formula(spec):
    horizon, n_paths = 500, 50_000
    found = numerical_kelly_fraction(spec, horizon=horizon, n_paths=n_paths, seed=0)

    # The only error is sampling noise in the win rate, scaled by 1 + 1/b.
    sd = (1 + 1 / spec.b) * math.sqrt(spec.p * spec.q / (horizon * n_paths))
    assert found == pytest.approx(kelly_fraction(spec), abs=6 * sd)


def test_the_simulated_optimum_is_reproducible():
    a = numerical_kelly_fraction(EDGE, horizon=100, n_paths=1_000, seed=7)
    b = numerical_kelly_fraction(EDGE, horizon=100, n_paths=1_000, seed=7)
    assert a == b


def test_the_simulated_optimum_without_an_edge_is_zero():
    found = numerical_kelly_fraction(BetSpec(p=0.4, b=1.0), horizon=200, n_paths=5_000, seed=0)
    assert found == pytest.approx(0.0, abs=1e-6)
