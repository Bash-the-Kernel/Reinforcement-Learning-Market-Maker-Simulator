"""Policy evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent import PPOAgent
from .environment import MarketMakerEnvironment


@dataclass(slots=True)
class EpisodeTrajectory:
    mid_prices: list[float]
    inventories: list[float]
    pnls: list[float]
    rewards: list[float]
    bid_quotes: list[float]
    ask_quotes: list[float]
    actions: list[np.ndarray]
    spread_capture: list[float]


def evaluate_policy(agent: PPOAgent, env: MarketMakerEnvironment, episodes: int = 5) -> tuple[list[dict], EpisodeTrajectory]:
    summaries: list[dict] = []
    trajectory = EpisodeTrajectory([], [], [], [], [], [], [], [])

    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        done = False
        episode_reward = 0.0
        episode_spread_capture = 0.0
        last_info: dict | None = None

        while not done:
            action, _, _ = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            last_info = info
            episode_spread_capture += info["spread_capture"]

            if episode == episodes - 1:
                trajectory.mid_prices.append(info["mid_price"])
                trajectory.inventories.append(info["inventory"])
                trajectory.pnls.append(info["mark_to_market"])
                trajectory.rewards.append(reward)
                trajectory.bid_quotes.append(info["bid_quote"])
                trajectory.ask_quotes.append(info["ask_quote"])
                trajectory.actions.append(np.asarray(action, dtype=np.float32))
                trajectory.spread_capture.append(info["spread_capture"])

        summaries.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "final_pnl": 0.0 if last_info is None else last_info["mark_to_market"],
                "final_inventory": 0.0 if last_info is None else last_info["inventory"],
                "average_spread_capture": episode_spread_capture / max(len(trajectory.rewards) if episode == episodes - 1 else env.config.episode_length, 1),
            }
        )

    return summaries, trajectory
