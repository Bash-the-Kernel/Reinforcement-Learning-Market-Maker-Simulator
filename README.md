# Reinforcement-Learning-Market-Maker-Simulator

Research-grade reinforcement learning market making simulator with:

- GBM mid-price dynamics
- simplified limit order book and Poisson market order arrivals
- discrete bid/ask quote control over a configurable offset grid
- Gym-style environment API with `reset()`, `step(action)`, `action_space`, and `observation_space`
- DQN training in PyTorch with experience replay, a target network, and epsilon-greedy exploration
- Avellaneda-Stoikov theoretical baseline for side-by-side comparison
- evaluation metrics and visualization plots

## Project Layout

- `src/rl_market_maker/environment.py` implements the market environment
- `src/rl_market_maker/agent.py` contains the DQN agent
- `src/rl_market_maker/baseline_agent.py` contains the Avellaneda-Stoikov baseline
- `src/rl_market_maker/comparison.py` runs agent-vs-baseline experiments and summary tables
- `src/rl_market_maker/training.py` runs training and evaluation
- `src/rl_market_maker/evaluation.py` collects episode trajectories
- `src/rl_market_maker/visualization.py` generates plots

## Run

Install dependencies from `pyproject.toml`, then run:

```bash
python -m rl_market_maker --episodes 250 --evaluation-episodes 5 --save-plots
```

Outputs are written to `artifacts/` by default, including a training reward curve and an episode dashboard.

To compare the DQN agent against the Avellaneda-Stoikov baseline, run:

```bash
python -m rl_market_maker --compare-agents --comparison-episodes 20 --save-plots
```

This generates a summary table in the terminal and comparison artifacts in `artifacts/comparison/`, including cumulative PnL curves and PnL distribution plots.

The environment observation is a 4D vector containing current mid price, agent inventory, spread, and short-term price momentum.
