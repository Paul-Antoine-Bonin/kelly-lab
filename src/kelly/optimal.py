"""Locating the optimal stake, numerically and in closed form.

The quantity being maximised is the expected log growth per bet,

    g(f) = p * log(1 + f*b) + q * log(1 - f)

which is what the long-run wealth of a single path converges to: after `n`
bets, log wealth divided by `n` tends to `g(f)` almost surely. Maximising the
*mean* of wealth instead would say "stake everything", which is how you go
broke with an edge.

Setting the derivative to zero gives the Kelly fraction

    f* = p - q / b = edge / b

clipped to `[0, 1]`: with no edge the best stake is nothing, and a bet that
cannot lose (`q == 0`) is staked in full.

This module finds the peak twice, independently, so that each checks the
other:

- `kelly_fraction` applies the formula above.
- `numerical_kelly_fraction` never sees the formula. It simulates a sweep of
  bets, measures the growth rate at whatever fractions a golden-section search
  asks for, and returns the argmax.

One thing worth knowing about the second: on a shared draw the simulated
growth rate is exactly `g(f)` with `p` replaced by the observed win rate,
because log wealth only depends on how many bets were won. So the simulated
peak sits at `p_hat - (1 - p_hat) / b`, and its error against the formula is
pure sampling noise in `p_hat`, of standard deviation
`(1 + 1/b) * sqrt(p*q / (n_paths * horizon))`. The search itself adds nothing
measurable on top.
"""

from __future__ import annotations

import math
from typing import Callable, Union

import numpy as np

from .simulate import BetSpec, SeedLike, as_generator

ArrayLike = Union[float, np.ndarray]

# 1 / golden ratio, the shrink factor of each golden-section step.
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def growth_rate(spec: BetSpec, fraction: ArrayLike) -> ArrayLike:
    """Expected log growth per bet, `p*log(1+fb) + q*log(1-f)`.

    Accepts a scalar or an array of fractions. At `fraction == 1` the loss
    term is `-inf`, and the result is `-inf` unless the bet cannot lose, in
    which case the `q * -inf` term is really zero rather than nan.
    """
    f = np.asarray(fraction, dtype=float)
    if np.any((f < 0.0) | (f > 1.0)):
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

    with np.errstate(divide="ignore"):
        log_win = np.log1p(f * spec.b)
        log_loss = np.log1p(-f)

    win_term = spec.p * log_win
    if spec.q == 0.0:
        g = win_term
    else:
        g = win_term + spec.q * log_loss
    return float(g) if g.ndim == 0 else g


def kelly_fraction(spec: BetSpec) -> float:
    """Closed-form optimal fraction, `p - q/b`, clipped to `[0, 1]`."""
    return float(min(max(spec.p - spec.q / spec.b, 0.0), 1.0))


def golden_section_max(
    func: Callable[[float], float],
    lo: float = 0.0,
    hi: float = 1.0,
    tol: float = 1e-7,
) -> float:
    """Argmax of a unimodal function on `[lo, hi]`, to within `tol`.

    Written out here rather than pulled from scipy, which the project does
    not otherwise need. The growth rate is concave in the fraction, so it is
    unimodal and golden-section search is guaranteed to converge on the peak.
    `-inf` values, which the growth rate takes at a fraction of 1, compare
    correctly and need no special case.
    """
    if not lo < hi:
        raise ValueError(f"need lo < hi, got [{lo}, {hi}]")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    a, b = lo, hi
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    fc, fd = func(c), func(d)
    while b - a > tol:
        # Keep the bracket that holds the larger interior value. On a tie,
        # either is valid for a unimodal function.
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = func(d)

    # The ends of the bracket are candidates too: a peak sitting on the
    # boundary (no edge, or a certain win) is approached but never evaluated
    # by the interior points alone.
    candidates = [(func(lo), lo), (func(hi), hi), (func((a + b) / 2), (a + b) / 2)]
    return max(candidates)[1]


def simulated_growth_curve(
    spec: BetSpec,
    horizon: int,
    n_paths: int,
    seed: SeedLike = None,
) -> Callable[[float], float]:
    """Mean simulated log growth per bet, as a function of the fraction.

    The bets are drawn once, here, and every fraction the returned function
    is asked about is evaluated on that same draw. Searching over fractions on
    fresh draws each time would make the curve jitter, and the search would
    chase the jitter instead of the peak.

    Only the win count per path matters for final log wealth, so the draw is
    one binomial per path, as in `simulate.final_log_wealth`.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths}")

    rng = as_generator(seed)
    wins = rng.binomial(horizon, spec.p, size=n_paths)
    losses = horizon - wins
    any_loss = bool(np.any(losses > 0))

    def curve(fraction: float) -> float:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
        log_win = math.log1p(fraction * spec.b)
        if fraction == 1.0:
            # A single loss anywhere puts that path, and so the mean, at -inf.
            # With no losses at all there is no loss term to speak of.
            if any_loss:
                return -math.inf
            return float(np.mean(wins)) * log_win / horizon
        log_loss = math.log1p(-fraction)
        total = wins * log_win + losses * log_loss
        return float(np.mean(total)) / horizon

    return curve


def numerical_kelly_fraction(
    spec: BetSpec,
    horizon: int = 500,
    n_paths: int = 50_000,
    seed: SeedLike = None,
    tol: float = 1e-7,
) -> float:
    """Optimal fraction found by simulation alone, without the formula.

    Draws `n_paths` runs of `horizon` bets once, then searches for the
    fraction that maximises their mean log growth.
    """
    curve = simulated_growth_curve(spec, horizon, n_paths, seed=seed)
    return golden_section_max(curve, 0.0, 1.0, tol=tol)
