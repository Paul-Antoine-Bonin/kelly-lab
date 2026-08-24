"""Simulation lab for the Kelly criterion and bet sizing.

`kelly.simulate` holds the repeated bet simulator: a bet described by a win
probability and net odds, staked at a fixed fraction of wealth over a horizon.
Everything accumulates in log wealth, so long horizons neither overflow nor
lose the exact zero that ruin produces.
"""

from .simulate import BetSpec

__version__ = "0.1.0"

__all__ = ["__version__", "BetSpec"]
