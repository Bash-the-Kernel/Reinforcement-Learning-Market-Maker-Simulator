"""Training loop for PPO market making."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .agent import PPOAgent, RolloutBuffer
from .config import TrainingConfig
from .environment import MarketMakerEnvironment
from .evaluation import evaluate_policy
from .visualization import plot_episode_dashboard, plot_training_curves


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
    agent = PPOAgent(env.observation_size, env.action_size, config.ppo, device=config.device)

    history: dict[str, list[float]] = {
        "episode": [],
        "episode_reward": [],
        "episode_pnl": [],
        "episode_inventory": [],
        "episode_spread_capture": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "smoothed_reward": [],
    }

    reward_window: list[float] = []
    rollout = RolloutBuffer()

    for episode in range(config.episodes):
        observation, _ = env.reset(seed=config.seed + episode)
        done = False
        episode_reward = 0.0
        episode_spread_capture = 0.0
        last_info: dict | None = None

        while not done:
            action, log_prob, value = agent.select_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rollout.add(observation, action, log_prob, reward, done, value)
            observation = next_observation
            episode_reward += reward
            episode_spread_capture += info["spread_capture"]
            last_info = info

        last_value = 0.0 if last_info is None or done else agent.select_action(observation, deterministic=True)[2]
        rollout.compute_returns_and_advantages(last_value, config.ppo.gamma, config.ppo.gae_lambda)
        update_metrics = agent.update(rollout.as_batch())
        rollout.clear()

        reward_window.append(episode_reward)
        if len(reward_window) > 20:
            reward_window.pop(0)

        history["episode"].append(episode)
        history["episode_reward"].append(episode_reward)
        history["episode_pnl"].append(0.0 if last_info is None else last_info["mark_to_market"])
        history["episode_inventory"].append(0.0 if last_info is None else last_info["inventory"])
        history["episode_spread_capture"].append(episode_spread_capture)
        history["smoothed_reward"].append(float(np.mean(reward_window)))
        history["policy_loss"].append(update_metrics["policy_loss"])
        history["value_loss"].append(update_metrics["value_loss"])
        history["entropy"].append(update_metrics["entropy"])

    evaluation_env = MarketMakerEnvironment(config.market)
    evaluation, trajectory = evaluate_policy(agent, evaluation_env, episodes=config.evaluation_episodes)

    if config.save_plots:
        plot_training_curves(history, output_dir / "training_rewards.png")
        plot_episode_dashboard(trajectory, output_dir / "episode_dashboard.png")

    return TrainingResult(history=history, evaluation=evaluation)
