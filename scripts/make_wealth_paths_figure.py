"""Regenerate figures/wealth-paths.png.

Run from the repo root:  python scripts/make_wealth_paths_figure.py

One run of 150 bets on a coin paying 2 to 1, staked at five fractions. The
optimal fraction for this bet is 0.25, which is why it is in the list.

The drawing floor is lowered to 1e-9 for this figure. At the module default of
1e-6 the three losing fractions all sit flat on the bottom and stop being
distinguishable, when the interesting part is how differently fast they fall.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from kelly.plots import plot_wealth_paths  # noqa: E402
from kelly.simulate import BetSpec  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "figures" / "wealth-paths.png"

BET = BetSpec(p=0.5, b=2.0)
FRACTIONS = [0.05, 0.25, 0.50, 0.75, 0.95]
HORIZON = 150
FLOOR = 1e-9
SEED = 0


def main() -> None:
    ax = plot_wealth_paths(
        BET, FRACTIONS, horizon=HORIZON, n_paths=1, seed=SEED, floor=FLOOR
    )
    OUT.parent.mkdir(exist_ok=True)
    ax.figure.tight_layout()
    ax.figure.savefig(OUT, dpi=150)
    plt.close(ax.figure)

    # No invented numbers in the README: print what was actually drawn.
    for fraction, line in zip(FRACTIONS, ax.lines):
        final = line.get_ydata()[-1]
        print(f"f = {fraction:.2f}   final wealth {final:.6g}")


if __name__ == "__main__":
    main()
