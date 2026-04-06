"""Visualization helpers for training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .evaluation import EpisodeTrajectory


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path | None = None):
    figure, axis = plt.subplots(figsize=(10, 5))
    episodes = history.get("episode", list(range(len(history.get("episode_reward", [])))))
    axis.plot(episodes, history.get("episode_reward", []), label="Episode reward", color="#0B6E99")
    if history.get("smoothed_reward"):
        axis.plot(episodes, history["smoothed_reward"], label="Smoothed reward", color="#D1495B")
    axis.set_title("Training Reward Curve")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Reward")
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return figure


def plot_episode_dashboard(trajectory: EpisodeTrajectory, output_path: str | Path | None = None):
    steps = range(len(trajectory.mid_prices))
    cumulative_pnl = []
    running = 0.0
    for reward in trajectory.rewards:
        running += reward
        cumulative_pnl.append(running)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(steps, trajectory.inventories, color="#7A5195")
    axes[0, 0].set_title("Inventory Over Time")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Inventory")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, cumulative_pnl, color="#0B6E99")
    axes[0, 1].set_title("Cumulative PnL")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("PnL")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, trajectory.mid_prices, label="Mid price", color="#333333")
    axes[1, 0].plot(steps, trajectory.bid_quotes, label="Bid quote", color="#2A9D8F", alpha=0.9)
    axes[1, 0].plot(steps, trajectory.ask_quotes, label="Ask quote", color="#E76F51", alpha=0.9)
    axes[1, 0].set_title("Quote Positions Relative to Price")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Price")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(steps, trajectory.spread_capture, color="#F4A261", width=1.0)
    axes[1, 1].set_title("Per-Step Spread Capture")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Spread Capture")
    axes[1, 1].grid(True, alpha=0.3)

    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return figure
