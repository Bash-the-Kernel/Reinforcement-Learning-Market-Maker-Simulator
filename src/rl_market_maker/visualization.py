"""Visualization helpers for training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def plot_training_stability(history: dict[str, list[float]], output_path: str | Path | None = None):
    figure, axis = plt.subplots(figsize=(10, 5))
    loss_values = np.asarray(history.get("loss", []), dtype=np.float32)
    if loss_values.size == 0:
        axis.text(0.5, 0.5, "No loss data available", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    else:
        valid_mask = np.isfinite(loss_values)
        filtered_loss = loss_values[valid_mask]
        filtered_steps = np.arange(len(loss_values))[valid_mask]
        axis.plot(filtered_steps, filtered_loss, color="#6A4C93", linewidth=1.6, label="Loss")
        if filtered_loss.size > 5:
            window = min(20, filtered_loss.size)
            kernel = np.ones(window, dtype=np.float32) / window
            smoothed = np.convolve(filtered_loss, kernel, mode="valid")
            smoothed_steps = filtered_steps[window - 1 :]
            axis.plot(smoothed_steps, smoothed, color="#D1495B", linewidth=2.0, label="Smoothed loss")
        axis.set_title("Training Stability")
        axis.set_xlabel("Update step")
        axis.set_ylabel("Loss")
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False)

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return figure


def _quoted_spread_bps(trajectory: EpisodeTrajectory) -> np.ndarray:
    if not trajectory.mid_prices:
        return np.asarray([], dtype=np.float32)

    mid_prices = np.asarray(trajectory.mid_prices, dtype=np.float32)
    bid_quotes = np.asarray(trajectory.bid_quotes, dtype=np.float32)
    ask_quotes = np.asarray(trajectory.ask_quotes, dtype=np.float32)
    return (ask_quotes - bid_quotes) / np.maximum(mid_prices, 1e-12) * 10_000.0


def plot_evaluation_dashboard(
    trajectory: EpisodeTrajectory,
    history: dict[str, list[float]] | None = None,
    output_path: str | Path | None = None,
):
    steps = np.arange(len(trajectory.mid_prices))
    cumulative_pnl = np.asarray(trajectory.pnls, dtype=np.float32)
    spread_bps = _quoted_spread_bps(trajectory)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(steps, cumulative_pnl, color="#0B6E99", linewidth=2)
    axes[0, 0].set_title("Cumulative PnL")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("PnL")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, trajectory.inventories, color="#7A5195", linewidth=2)
    axes[0, 1].set_title("Inventory Position Over Time")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Inventory")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(spread_bps, bins=min(max(len(spread_bps) // 10, 10), 50), color="#F4A261", edgecolor="#FFFFFF", alpha=0.9)
    axes[1, 0].set_title("Distribution of Quoted Spreads")
    axes[1, 0].set_xlabel("Spread (bps)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, alpha=0.25)

    if history is not None:
        episodes = history.get("episode", list(range(len(history.get("episode_reward", [])))))
        axes[1, 1].plot(episodes, history.get("episode_reward", []), label="Episode reward", color="#0B6E99")
        if history.get("smoothed_reward"):
            axes[1, 1].plot(episodes, history["smoothed_reward"], label="Smoothed reward", color="#D1495B")
        axes[1, 1].set_title("Training Reward Curve")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Reward")
        axes[1, 1].legend(frameon=False)
    else:
        axes[1, 1].plot(steps, trajectory.bid_quotes, label="Bid quote", color="#2A9D8F", alpha=0.9)
        axes[1, 1].plot(steps, trajectory.ask_quotes, label="Ask quote", color="#E76F51", alpha=0.9)
        axes[1, 1].set_title("Quoted Bid and Ask Prices")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Price")
        axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)

    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return figure


def plot_episode_dashboard(trajectory: EpisodeTrajectory, output_path: str | Path | None = None):
    return plot_evaluation_dashboard(trajectory, history=None, output_path=output_path)
