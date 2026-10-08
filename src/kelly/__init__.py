"""Simulation lab for the Kelly criterion and bet sizing.

`kelly.simulate` holds the repeated bet simulator: a bet described by a win
probability and net odds, staked at a fixed fraction of wealth over a horizon.
Everything accumulates in log wealth, so long horizons neither overflow nor
lose the exact zero that ruin produces.

`kelly.optimal` locates the best stake twice: the closed-form Kelly fraction,
and a search over simulated growth rates that never sees the formula.

`kelly.plots` draws the results. It is imported on demand rather than here, so
that using the simulator never pulls matplotlib in.
"""

from .simulate import BetSpec

__version__ = "0.1.0"

__all__ = ["__version__", "BetSpec"]
