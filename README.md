# kelly-lab

Why betting too big ruins you even when you have an edge.

With a real, positive edge, staking too large a fraction of your wealth leads to
near-certain ruin. This repo builds the simulations that show it.

The simulator works. The charts and the optimal fraction come next.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## The simulator

```python
from kelly.simulate import BetSpec, wealth_paths, final_log_wealth, ruin_probability

bet = BetSpec(p=0.5, b=2.0)          # a fair coin paying 2 to 1
bet.edge                             # +0.50 profit per unit staked

wealth_paths(bet, fraction=0.25, horizon=200, n_paths=5, seed=0)
final_log_wealth(bet, fraction=0.25, horizon=500, n_paths=50_000, seed=0)
ruin_probability(bet, fraction=0.80, horizon=500, n_paths=50_000, seed=0)
```

Four parameters: the win probability `p`, the net odds `b`, the fraction of wealth
staked on every bet, and the horizon. A win multiplies wealth by `1 + fraction * b`,
a loss by `1 - fraction`, and the two compose.

Everything accumulates in **log wealth**. Over a long horizon the natural scale
overflows float64 well before the simulation stops being interesting, and the sum of
logs does not. `wealth_paths` exponentiates on the way out when you want the raw
numbers.

Ruin is exact. Staking the whole bankroll turns one loss into wealth of precisely
zero, whose log is `-inf`, and the running sum keeps it there. There is no recovery
from zero and the simulator does not pretend otherwise.

## Verified against the closed form

Expected log growth per bet is `p·log(1+fb) + q·log(1-f)`. Simulating 50,000 paths of
500 bets reproduces it:

| Fraction | Analytic g | Simulated g | Error | Runs left with under 1% |
|---|---|---|---|---|
| 0.05 | 0.02201 | 0.02201 | 0.00000 | 0.0% |
| 0.10 | 0.03848 | 0.03849 | 0.00001 | 0.0% |
| 0.25 | 0.05889 | 0.05891 | 0.00002 | 0.0% |
| 0.40 | 0.03848 | 0.03852 | 0.00004 | 2.7% |
| 0.60 | -0.06392 | -0.06386 | 0.00006 | 93.0% |
| 0.80 | -0.32696 | -0.32688 | 0.00008 | 100.0% |
| 0.95 | -0.96551 | -0.96538 | 0.00013 | 100.0% |

Every row is the same bet with the same +0.50 edge. Only the stake changes. By 0.60
the growth rate is negative and 93% of runs have lost 99% of the bankroll, and none of
that comes from a bad bet. That is the whole thesis of the repo, and locating the peak
and drawing the curve are the next steps in [TODO.md](TODO.md).

## Speed

The win count over a horizon is binomial, so a final wealth only needs one draw per
path rather than one per bet. Same distribution, and for 50,000 paths of 500 bets:

| | |
|---|---|
| `final_log_wealth`, binomial | 6.2 ms |
| `simulate_log_paths`, full history | 3.3 s |

Use the full paths when the shape of the path is the thing being plotted, and the fast
one for sweeps.

## Run the tests

```bash
pytest
```

40 tests. The deterministic ones pin the mechanics: a certain win compounds at exactly
`1 + fb`, wealth only ever moves by one of two factors, ruin never recovers. Three
statistical ones check the simulator against the closed form on a fixed seed.

Roadmap and progress: [TODO.md](TODO.md)
