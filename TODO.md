# kelly-lab: why betting too big ruins you even when you have an edge

## Goal
Build a simulation lab that shows visually the most counter-intuitive result in risk
management: with a real, positive edge, betting too big leads to near-certain ruin.
It is the trading floor's favourite topic.

## Data
None. Pure simulation, which is an advantage: nothing can block the project.

## Steps
- [x] Set up repo, `src/` layout plus a demonstration notebook
- [x] Repeated bet simulator: win probability, odds, fraction staked, horizon
- [x] Plot wealth paths for several fractions, same random seed
- [x] Recover the optimal Kelly fraction numerically, then check it against the
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
- Fix the random seed so the figures are reproducible. Every entry point takes one and
  none of them touch the global numpy state, so a figure cannot change because
  something else drew a random number first.
- Mean final wealth can rise while the median collapses. Never show the mean alone.
- Accumulate in log wealth, not by multiplying. A few hundred bets at a healthy growth
  rate overflow float64 on the natural scale, and the interesting horizons are longer
  than that.
- Ruin has to be exactly zero, never a small positive number. At a fraction of 1 a
  single loss multiplies wealth by 0, whose log is -inf, and the running sum must keep
  it there. Watch the `0 * -inf` case: no losses at fraction 1 is nan if computed
  naively, when the answer is simply no losses.
- Comparing fractions only means anything on one shared draw. `draw_wins` produces the
  outcomes once and `log_paths_from_wins` replays them at each fraction; simulating
  each fraction separately would mix sizing up with luck.
- A logarithmic wealth axis cannot draw the exact zero that ruin produces. `plots`
  floors the drawn value instead, which is a drawing convention and not a result.
- Drawing the win count from a binomial rather than stepping bet by bet gives the same
  distribution about 500 times faster (6 ms against 3.3 s for 50,000 paths of 500
  bets). Use the full paths only when the shape of the path is what you are plotting.
