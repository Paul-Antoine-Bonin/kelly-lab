"""Repeated bet simulator.

One bet is described by four numbers, and the whole project is about what
happens when you vary the last two:

- `p`         probability of winning
- `b`         net odds: a winning unit stake returns `b` on top of the stake
- `fraction`  the share of current wealth staked on every bet
- `horizon`   how many bets in a row

Wealth is multiplicative. A win multiplies it by `1 + fraction * b`, a loss by
`1 - fraction`, and the two compose over the horizon. That is the entire
mechanism behind the result this repo exists to show: an edge tells you the
sign of each bet, but it says nothing about the size that survives compounding.

Everything is accumulated in **log wealth**. Over ten thousand bets a
multiplicative path overflows float64 long before the simulation is
interesting, and the sum of logs does not. `wealth_paths` exponentiates on the
way out for callers who want the raw scale.

Ruin is exact rather than approximate. Staking the whole bankroll, `fraction`
of 1, turns a single loss into wealth of exactly zero, whose log is `-inf`, and
the cumulative sum keeps it there forever. That is the correct behaviour: there
is no recovery from zero, and the simulator must not quietly round it to a
small positive number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

SeedLike = Union[int, np.random.Generator, None]


@dataclass(frozen=True)
class BetSpec:
    """A repeated bet: win with probability `p`, net odds `b`.

    Decimal odds `d` quoted by a bookmaker correspond to `b = d - 1`, since the
    decimal figure includes the returned stake and `b` does not.
    """

    p: float
    b: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be a probability, got {self.p}")
        if self.b <= 0.0:
            raise ValueError(f"net odds b must be positive, got {self.b}")

    @property
    def q(self) -> float:
        """Probability of losing."""
        return 1.0 - self.p

    @property
    def edge(self) -> float:
        """Expected profit per unit staked, `p*b - q`.

        Positive means the bet is worth taking at *some* size. It says nothing
        about which size, which is the point of the whole repo.
        """
        return self.p * self.b - self.q

    @property
    def has_edge(self) -> bool:
        return self.edge > 0.0


def as_generator(seed: SeedLike = None) -> np.random.Generator:
    """Accept a seed, an existing Generator, or nothing.

    Figures in this repo have to be reproducible, so every entry point takes a
    seed and none of them touches the global numpy random state.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def draw_wins(
    spec: BetSpec,
    horizon: int,
    n_paths: int = 1,
    seed: SeedLike = None,
) -> np.ndarray:
    """Win/loss outcomes, shape `(n_paths, horizon)`, `True` for a win.

    Split out from the path builder so that several fractions can be replayed
    on the *same* coin flips. Comparing stake sizes on independent draws is
    comparing luck as much as sizing, and the whole point of the comparison is
    that the sequence of bets is not what separates them.
    """
    rng = as_generator(seed)
    return rng.random((n_paths, horizon)) < spec.p


def log_paths_from_wins(
    wins: np.ndarray,
    spec: BetSpec,
    fraction: float,
) -> np.ndarray:
    """Turn win/loss outcomes into log wealth paths at a given `fraction`.

    `wins` has shape `(n_paths, horizon)`; the result has one extra column at
    the front for the starting bankroll, at log wealth 0.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"fraction must be between 0 and 1, got {fraction}. Above 1 would "
            f"mean staking borrowed money, which this simulator does not model."
        )
    wins = np.atleast_2d(np.asarray(wins, dtype=bool))
    n_paths, horizon = wins.shape

    # log1p keeps precision for the small fractions that matter most, and
    # errstate silences the deliberate log(0) at fraction == 1.
    with np.errstate(divide="ignore"):
        log_win = np.log1p(fraction * spec.b)
        log_loss = np.log1p(-fraction)

    steps = np.where(wins, log_win, log_loss)
    paths = np.zeros((n_paths, horizon + 1), dtype=float)
    np.cumsum(steps, axis=1, out=paths[:, 1:])
    return paths


def simulate_log_paths(
    spec: BetSpec,
    fraction: float,
    horizon: int,
    n_paths: int = 1,
    seed: SeedLike = None,
) -> np.ndarray:
    """Log wealth paths, shape `(n_paths, horizon + 1)`, all starting at 0.

    Column 0 is the starting bankroll, so a horizon of `n` gives `n + 1`
    columns. Log wealth of `-inf` means ruin, and once reached it persists.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            f"fraction must be between 0 and 1, got {fraction}. Above 1 would "
            f"mean staking borrowed money, which this simulator does not model."
        )
    if horizon < 0:
        raise ValueError(f"horizon must not be negative, got {horizon}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths}")

    wins = draw_wins(spec, horizon, n_paths=n_paths, seed=seed)
    return log_paths_from_wins(wins, spec, fraction)


def wealth_paths(
    spec: BetSpec,
    fraction: float,
    horizon: int,
    n_paths: int = 1,
    seed: SeedLike = None,
) -> np.ndarray:
    """Wealth paths on the natural scale, starting at 1.

    Convenient for plotting a handful of paths. For long horizons prefer
    `simulate_log_paths`: this one overflows to `inf` where the log scale stays
    finite and readable.
    """
    return np.exp(
        simulate_log_paths(spec, fraction, horizon, n_paths=n_paths, seed=seed)
    )


def final_log_wealth(
    spec: BetSpec,
    fraction: float,
    horizon: int,
    n_paths: int = 1,
    seed: SeedLike = None,
) -> np.ndarray:
    """Log wealth after `horizon` bets, one value per path.

    Allocates only the current column rather than the whole history, so a
    sweep over many fractions with many paths stays cheap.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    if horizon < 0:
        raise ValueError(f"horizon must not be negative, got {horizon}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths}")

    rng = as_generator(seed)
    with np.errstate(divide="ignore"):
        log_win = np.log1p(fraction * spec.b)
        log_loss = np.log1p(-fraction)

    if horizon == 0:
        return np.zeros(n_paths)

    # Number of wins is binomial, so the whole horizon collapses to one draw
    # per path. Identical in distribution to stepping bet by bet, and it makes
    # horizons of millions practical.
    wins = rng.binomial(horizon, spec.p, size=n_paths)
    losses = horizon - wins

    with np.errstate(invalid="ignore"):
        total = wins * log_win + losses * log_loss

    # 0 losses at fraction == 1 would give 0 * -inf = nan, which is really 0.
    if np.isneginf(log_loss):
        total = np.where(losses == 0, wins * log_win, -np.inf)
    return total


def ruin_probability(
    spec: BetSpec,
    fraction: float,
    horizon: int,
    n_paths: int = 10_000,
    threshold: float = 0.01,
    seed: SeedLike = None,
) -> float:
    """Share of paths ending below `threshold` times the starting bankroll.

    Total ruin only happens at `fraction` of exactly 1. Everywhere else wealth
    approaches zero without reaching it, so "ruin" needs a threshold, and the
    honest thing is to make the caller see it.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    final = final_log_wealth(
        spec, fraction, horizon, n_paths=n_paths, seed=seed
    )
    return float(np.mean(final < np.log(threshold)))
