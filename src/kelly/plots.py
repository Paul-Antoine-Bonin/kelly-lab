"""Charts for the simulation lab.

The first chart of the repo, and the one that makes the argument on its own:
the same run of coin flips, staked at several different fractions, drawn on one
pair of axes.

Everything here takes the outcomes from a single `draw_wins` call and replays
them at each fraction. That is deliberate. Simulating each fraction against its
own independent flips would leave the reader unable to tell sizing apart from
luck, and the point being made is precisely that the flips are not what
separates the curves.

Wealth is drawn on a logarithmic axis. Multiplicative growth is a straight line
there, which is what makes a growth *rate* something the eye can read, and the
alternative hides everything below the top path behind the top path.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .simulate import BetSpec, SeedLike, draw_wins, log_paths_from_wins

# Ruined paths reach wealth of exactly zero, which no logarithmic axis can
# show. They are drawn down to this floor instead and stop there, so that a
# line falling off the bottom of the chart reads as what it is.
RUIN_FLOOR = 1e-6


def plot_wealth_paths(
    spec: BetSpec,
    fractions: Sequence[float],
    horizon: int = 250,
    n_paths: int = 1,
    seed: SeedLike = 0,
    ax=None,
    floor: float = RUIN_FLOOR,
):
    """Draw wealth against bet number, one colour per fraction.

    All fractions share one set of coin flips, so any difference between the
    curves comes from the stake and nothing else. With `n_paths` above 1 every
    fraction gets the same `n_paths` runs, drawn faded.

    Returns the matplotlib `Axes`. Nothing is saved and nothing is shown; the
    caller decides.
    """
    import matplotlib.pyplot as plt

    fractions = list(fractions)
    if not fractions:
        raise ValueError("give at least one fraction to plot")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if floor <= 0.0:
        raise ValueError(f"floor must be positive, got {floor}")

    wins = draw_wins(spec, horizon, n_paths=n_paths, seed=seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))

    bets = np.arange(horizon + 1)
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(fractions)))

    for fraction, colour in zip(fractions, colours):
        wealth = np.exp(log_paths_from_wins(wins, spec, fraction))
        # Clip only for display: the zeros are real, the floor is a drawing
        # convention, and clipping upwards would never move a live path.
        wealth = np.maximum(wealth, floor)
        for i, path in enumerate(wealth):
            ax.plot(
                bets,
                path,
                color=colour,
                linewidth=1.6 if n_paths == 1 else 1.0,
                alpha=1.0 if n_paths == 1 else 0.55,
                label=f"f = {fraction:.2f}" if i == 0 else None,
            )

    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("Bet number")
    ax.set_ylabel("Wealth, starting bankroll = 1")
    ax.set_title(
        f"One run of {horizon} bets at p = {spec.p:g}, b = {spec.b:g} "
        f"(edge {spec.edge:+.2f}), staked at different fractions"
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="major", alpha=0.25)
    return ax
