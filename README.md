# kelly-lab

Why betting too big ruins you even when you have an edge.

With a real, positive edge, staking too large a fraction of your wealth leads to
near-certain ruin. This repo builds the simulations that show it.

No results yet. The simulator is the next step.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run the tests

```bash
pytest
```

## Layout

```
src/kelly/      the package
tests/          test suite
notebooks/      01-kelly-demo.ipynb, the readable walkthrough
```

Roadmap and progress: [TODO.md](TODO.md)
