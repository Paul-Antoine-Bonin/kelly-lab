# kelly-lab: why betting too big ruins you even when you have an edge

## Goal
Build a simulation lab that shows visually the most counter-intuitive result in risk
management: with a real, positive edge, betting too big leads to near-certain ruin.
It is the trading floor's favourite topic.

## Data
None. Pure simulation, which is an advantage: nothing can block the project.

## Steps
- [x] Set up repo, `src/` layout plus a demonstration notebook
- [ ] Repeated bet simulator: win probability, odds, fraction staked, horizon
- [ ] Plot wealth paths for several fractions, same random seed
- [ ] Recover the optimal Kelly fraction numerically, then check it against the
      analytical formula
- [ ] Show the bell curve of growth rate against fraction, and the point where it
      drops back below zero despite a positive edge
- [ ] Separate mean and median final wealth: the gap is the heart of the subject
- [ ] Add fractional Kelly (half Kelly) and measure the growth against drawdown tradeoff
- [ ] Case where the edge is estimated with error: show how sensitive Kelly is to
      overestimating the edge
- [ ] Extension: several correlated simultaneous bets
- [ ] README: the three charts that tell the whole story

## Done when
A notebook a non-specialist can run through and understand in ten minutes, and a
README with the key charts.

## Traps
- Fix the random seed so the figures are reproducible.
- Mean final wealth can rise while the median collapses. Never show the mean alone.
