# Reinforcement-Learning-Market-Maker-Simulator

Research-grade reinforcement learning market making simulator with:

- GBM mid-price dynamics
- simplified limit order book and Poisson market order arrivals
- continuous bid/ask quote control
- PPO training in PyTorch
- evaluation metrics and visualization plots

## Project Layout

- `src/rl_market_maker/environment.py` implements the market environment
- `src/rl_market_maker/agent.py` contains the PPO actor-critic agent
- `src/rl_market_maker/training.py` runs training and evaluation
- `src/rl_market_maker/evaluation.py` collects episode trajectories
- `src/rl_market_maker/visualization.py` generates plots

## Run

Install dependencies from `pyproject.toml`, then run:

```bash
python -m rl_market_maker --episodes 250 --evaluation-episodes 5 --save-plots
```

Outputs are written to `artifacts/` by default, including a training reward curve and an episode dashboard.
