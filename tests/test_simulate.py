"""Tests for the repeated bet simulator.

Deterministic wherever possible. The few statistical checks use a fixed seed
and a tolerance wide enough that they cannot flake, but narrow enough to catch
a real error.
"""

import numpy as np
import pytest

from kelly.simulate import (
    BetSpec,
    as_generator,
    final_log_wealth,
    ruin_probability,
    simulate_log_paths,
    wealth_paths,
)

# A coin flip paying 2 to 1: clear positive edge, easy arithmetic.
EDGE = BetSpec(p=0.5, b=2.0)
FAIR = BetSpec(p=0.5, b=1.0)


# --- the bet description -----------------------------------------------------


def test_edge_is_expected_profit_per_unit():
    # 0.5 * 2 - 0.5 = 0.5
    assert EDGE.edge == pytest.approx(0.5)
    assert EDGE.has_edge


def test_a_fair_coin_at_evens_has_no_edge():
    assert FAIR.edge == pytest.approx(0.0)
    assert not FAIR.has_edge


def test_a_negative_edge_is_reported_as_such():
    assert BetSpec(p=0.4, b=1.0).edge == pytest.approx(-0.2)
    assert not BetSpec(p=0.4, b=1.0).has_edge


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_p_must_be_a_probability(p):
    with pytest.raises(ValueError):
        BetSpec(p=p, b=2.0)


@pytest.mark.parametrize("b", [0.0, -1.0])
def test_net_odds_must_be_positive(b):
    with pytest.raises(ValueError):
        BetSpec(p=0.5, b=b)


def test_q_complements_p():
    assert BetSpec(p=0.6, b=2.0).q == pytest.approx(0.4)


# --- shapes and validation ---------------------------------------------------


def test_path_shape_includes_the_starting_point():
    paths = simulate_log_paths(EDGE, 0.25, horizon=10, n_paths=7, seed=0)
    assert paths.shape == (7, 11)


def test_paths_start_at_zero_log_wealth():
    paths = simulate_log_paths(EDGE, 0.25, horizon=5, n_paths=3, seed=0)
    assert np.all(paths[:, 0] == 0.0)


def test_wealth_paths_start_at_one():
    assert np.all(wealth_paths(EDGE, 0.25, horizon=5, n_paths=3, seed=0)[:, 0] == 1.0)


def test_a_zero_horizon_is_just_the_starting_bankroll():
    assert simulate_log_paths(EDGE, 0.25, horizon=0, n_paths=4, seed=0).shape == (4, 1)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, 2.0])
def test_fraction_outside_zero_to_one_is_refused(fraction):
    with pytest.raises(ValueError):
        simulate_log_paths(EDGE, fraction, horizon=5, seed=0)


def test_the_fraction_error_explains_why():
    with pytest.raises(ValueError) as exc:
        simulate_log_paths(EDGE, 1.5, horizon=5, seed=0)
    assert "borrowed" in str(exc.value)


def test_negative_horizon_is_refused():
    with pytest.raises(ValueError):
        simulate_log_paths(EDGE, 0.25, horizon=-1, seed=0)


def test_n_paths_must_be_at_least_one():
    with pytest.raises(ValueError):
        simulate_log_paths(EDGE, 0.25, horizon=5, n_paths=0, seed=0)


# --- determinism -------------------------------------------------------------


def test_the_same_seed_gives_the_same_paths():
    a = simulate_log_paths(EDGE, 0.3, horizon=50, n_paths=5, seed=42)
    b = simulate_log_paths(EDGE, 0.3, horizon=50, n_paths=5, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_paths():
    a = simulate_log_paths(EDGE, 0.3, horizon=50, n_paths=5, seed=1)
    b = simulate_log_paths(EDGE, 0.3, horizon=50, n_paths=5, seed=2)
    assert not np.array_equal(a, b)


def test_the_global_numpy_state_is_untouched():
    # Figures must not change because something else drew a random number.
    np.random.seed(0)
    before = np.random.get_state()[2]
    simulate_log_paths(EDGE, 0.3, horizon=100, n_paths=10, seed=7)
    assert np.random.get_state()[2] == before


def test_a_generator_can_be_passed_instead_of_a_seed():
    paths = simulate_log_paths(EDGE, 0.3, horizon=5, seed=np.random.default_rng(3))
    assert paths.shape == (1, 6)


def test_as_generator_passes_a_generator_through():
    rng = np.random.default_rng(0)
    assert as_generator(rng) is rng


# --- the mechanics of one bet ------------------------------------------------


def test_staking_nothing_never_changes_wealth():
    paths = simulate_log_paths(EDGE, 0.0, horizon=100, n_paths=10, seed=0)
    assert np.all(paths == 0.0)


def test_a_certain_win_compounds_at_the_win_factor():
    certain = BetSpec(p=1.0, b=2.0)
    wealth = wealth_paths(certain, 0.5, horizon=3, seed=0)[0]
    # every bet multiplies by 1 + 0.5*2 = 2
    assert wealth == pytest.approx([1.0, 2.0, 4.0, 8.0])


def test_a_certain_loss_compounds_at_the_loss_factor():
    doomed = BetSpec(p=0.0, b=2.0)
    wealth = wealth_paths(doomed, 0.5, horizon=3, seed=0)[0]
    # every bet multiplies by 1 - 0.5 = 0.5
    assert wealth == pytest.approx([1.0, 0.5, 0.25, 0.125])


def test_wealth_only_ever_moves_by_one_of_two_factors():
    spec = BetSpec(p=0.5, b=2.0)
    paths = wealth_paths(spec, 0.25, horizon=200, n_paths=5, seed=11)
    ratios = np.unique(np.round(paths[:, 1:] / paths[:, :-1], 12))
    assert set(ratios) == {0.75, 1.5}


# --- ruin is exact -----------------------------------------------------------


def test_staking_everything_and_losing_is_absorbing():
    doomed = BetSpec(p=0.0, b=2.0)
    paths = simulate_log_paths(doomed, 1.0, horizon=5, seed=0)
    assert np.all(np.isneginf(paths[:, 1:]))
    assert np.all(wealth_paths(doomed, 1.0, horizon=5, seed=0)[:, 1:] == 0.0)


def test_ruin_never_recovers():
    # One loss anywhere at fraction 1 must leave every later value at zero.
    paths = simulate_log_paths(BetSpec(p=0.5, b=2.0), 1.0, horizon=30,
                               n_paths=50, seed=5)
    ruined = np.isneginf(paths)
    # once True along a row, always True
    assert np.all(np.maximum.accumulate(ruined, axis=1) == ruined)


def test_staking_everything_with_a_certain_win_is_not_ruin():
    certain = BetSpec(p=1.0, b=2.0)
    assert np.all(np.isfinite(simulate_log_paths(certain, 1.0, horizon=10, seed=0)))


def test_final_log_wealth_handles_the_zero_loss_case_at_fraction_one():
    # 0 losses times -inf is nan if computed naively; it must be 0 losses.
    certain = BetSpec(p=1.0, b=2.0)
    final = final_log_wealth(certain, 1.0, horizon=4, n_paths=3, seed=0)
    assert np.all(np.isfinite(final))
    assert final == pytest.approx(np.full(3, 4 * np.log(3.0)))


# --- the fast path agrees with the slow one ----------------------------------


def test_final_log_wealth_matches_the_end_of_a_full_path():
    # Same distribution, not the same draws, so compare summary statistics.
    stepwise = simulate_log_paths(EDGE, 0.25, horizon=200, n_paths=20_000, seed=1)[:, -1]
    binomial = final_log_wealth(EDGE, 0.25, horizon=200, n_paths=20_000, seed=2)
    assert np.mean(binomial) == pytest.approx(np.mean(stepwise), rel=0.02)
    assert np.std(binomial) == pytest.approx(np.std(stepwise), rel=0.05)


def test_mean_log_growth_matches_the_analytic_value():
    # E[log growth] per bet = p*log(1+fb) + q*log(1-f), by construction.
    fraction = 0.2
    expected = 0.5 * np.log(1 + fraction * 2.0) + 0.5 * np.log(1 - fraction)
    final = final_log_wealth(EDGE, fraction, horizon=500, n_paths=20_000, seed=3)
    assert np.mean(final) / 500 == pytest.approx(expected, abs=2e-3)


def test_win_count_matches_the_win_probability():
    spec = BetSpec(p=0.3, b=2.0)
    paths = simulate_log_paths(spec, 0.1, horizon=1000, n_paths=200, seed=4)
    steps = np.diff(paths, axis=1)
    observed = np.mean(steps > 0)
    assert observed == pytest.approx(0.3, abs=0.01)


# --- ruin probability --------------------------------------------------------


def test_ruin_probability_is_zero_when_nothing_is_staked():
    assert ruin_probability(EDGE, 0.0, horizon=100, n_paths=500, seed=0) == 0.0


def test_ruin_probability_rises_with_the_fraction():
    # The headline result: same positive edge, bigger stake, worse outcome.
    small = ruin_probability(EDGE, 0.1, horizon=300, n_paths=4000, seed=0)
    large = ruin_probability(EDGE, 0.8, horizon=300, n_paths=4000, seed=0)
    assert small < large
    assert large > 0.9


def test_ruin_threshold_must_be_positive():
    with pytest.raises(ValueError):
        ruin_probability(EDGE, 0.5, horizon=10, threshold=0.0, seed=0)
