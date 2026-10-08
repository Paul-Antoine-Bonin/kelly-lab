"""Recover the optimal fraction by simulation and compare it with the formula.

Run from the repo root:  python scripts/check_kelly_fraction.py

For each bet, 50,000 runs of 500 bets are drawn once and a golden-section
search finds the fraction with the highest mean log growth. That search never
sees `p - q/b`. The table printed here is the one in the README, so the
README carries no number this script did not produce.
"""

import math

from kelly.optimal import growth_rate, kelly_fraction, numerical_kelly_fraction
from kelly.simulate import BetSpec

BETS = [
    BetSpec(p=0.5, b=2.0),
    BetSpec(p=0.6, b=1.0),
    BetSpec(p=0.55, b=1.0),
    BetSpec(p=0.3, b=4.0),
    BetSpec(p=0.1, b=12.0),
    BetSpec(p=0.9, b=0.2),
]
HORIZON = 500
N_PATHS = 50_000
SEED = 0


def main() -> None:
    print("| p | b | Edge | Formula f* | Simulated f* | Error | Sampling sd | g at f* |")
    print("|---|---|---|---|---|---|---|---|")
    for spec in BETS:
        exact = kelly_fraction(spec)
        found = numerical_kelly_fraction(
            spec, horizon=HORIZON, n_paths=N_PATHS, seed=SEED
        )
        sd = (1 + 1 / spec.b) * math.sqrt(spec.p * spec.q / (HORIZON * N_PATHS))
        print(
            f"| {spec.p:g} | {spec.b:g} | {spec.edge:+.3f} | {exact:.5f} "
            f"| {found:.5f} | {found - exact:+.5f} | {sd:.5f} "
            f"| {growth_rate(spec, exact):.5f} |"
        )


if __name__ == "__main__":
    main()
