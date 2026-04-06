"""Training loop for DQN market making."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .agent import DQNAgent
from .config import TrainingConfig
from .environment import MarketMakerEnvironment
from .evaluation import evaluate_policy
from .visualization import plot_evaluation_dashboard, plot_training_curves, plot_training_stability


@dataclass(slots=True)
class TrainingResult:
    history: dict[str, list[float]]
    evaluation: list[dict]


def train_agent(config: TrainingConfig | None = None) -> TrainingResult:
    config = config or TrainingConfig()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = MarketMakerEnvironment(config.market)
    agent = DQNAgent(env.observation_size, env.action_grid, config.dqn, device=config.device)

    history: dict[str, list[float]] = {
        "episode": [],
        "episode_reward": [],
        "episode_total_pnl": [],
        "episode_average_inventory": [],
        "episode_spread_capture": [],
        "loss": [],
        "epsilon": [],
        "smoothed_reward": [],
    }

    reward_window: list[float] = []
    total_steps = 0

    for episode in range(config.episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        done = False
        episode_reward = 0.0
        episode_spread_capture = 0.0
        inventory_sum = 0.0
        inventory_steps = 0
        last_info: dict | None = None

        while not done:
            action_index, action, epsilon = agent.select_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action_index)
            done = terminated or truncated

            agent.add_transition(observation, action_index, reward, next_observation, done)
            observation = next_observation
            episode_reward += reward
            episode_spread_capture += info["spread_capture"]
            inventory_sum += float(info["inventory"])
            inventory_steps += 1
            last_info = info
            total_steps += 1

            if total_steps % config.dqn.update_interval == 0:
                update_metrics = agent.update()
            else:
                update_metrics = None

        reward_window.append(episode_reward)
        if len(reward_window) > 20:
            reward_window.pop(0)

        history["episode"].append(episode)
        history["episode_reward"].append(episode_reward)
        history["episode_total_pnl"].append(0.0 if last_info is None else last_info["mark_to_market"])
        history["episode_average_inventory"].append(inventory_sum / max(inventory_steps, 1))
        history["episode_spread_capture"].append(episode_spread_capture)
        history["smoothed_reward"].append(float(np.mean(reward_window)))
        if update_metrics is not None:
            history["loss"].append(update_metrics["loss"])
            history["epsilon"].append(update_metrics["epsilon"])
        else:
            history["loss"].append(float("nan"))
            history["epsilon"].append(agent.epsilon())

    agent.sync_target_network()

    evaluation_env = MarketMakerEnvironment(config.market)
    evaluation, trajectory = evaluate_policy(agent, evaluation_env, episodes=config.evaluation_episodes)

    if config.save_plots:
        plot_training_curves(history, output_dir / "training_rewards.png")
        plot_training_stability(history, output_dir / "training_stability.png")
        plot_evaluation_dashboard(trajectory, history=history, output_path=output_dir / "episode_dashboard.png")

    return TrainingResult(history=history, evaluation=evaluation)
