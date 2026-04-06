"""Experiment framework for comparing the DQN agent with the Avellaneda-Stoikov baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .agent import DQNAgent
from .baseline_agent import AvellanedaStoikovAgent
from .config import AvellanedaStoikovConfig, DQNConfig, MarketMakerConfig, TrainingConfig
from .environment import MarketMakerEnvironment
from .visualization import plot_evaluation_dashboard, plot_training_curves


@dataclass(slots=True)
class AgentEpisodeResult:
    mid_prices: list[float]
    rewards: list[float]
    pnls: list[float]
    inventories: list[float]
    bid_quotes: list[float]
    ask_quotes: list[float]
    spread_capture: list[float]
    trade_pnls: list[float]
    drawdowns: list[float]
    inventory_variance: float
    total_pnl: float
    sharpe_ratio: float
    average_spread_captured: float
    trade_win_rate: float
    cumulative_pnl: list[float]


@dataclass(slots=True)
class ComparisonSummary:
    rows: list[dict[str, float | str]]
    dqn_result: AgentEpisodeResult
    baseline_result: AgentEpisodeResult
    output_dir: Path


def _compute_sharpe(rewards: list[float]) -> float:
    if len(rewards) < 2:
        return 0.0
    reward_array = np.asarray(rewards, dtype=np.float32)
    std = reward_array.std(ddof=1)
    if std <= 1e-12:
        return 0.0
    return float(reward_array.mean() / std * np.sqrt(len(reward_array)))


def _compute_drawdowns(cumulative_pnl: list[float]) -> list[float]:
    if not cumulative_pnl:
        return []
    wealth = np.asarray(cumulative_pnl, dtype=np.float32)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = wealth - running_max
    return drawdowns.tolist()


def _run_policy_episode(policy_name: str, env: MarketMakerEnvironment, policy_fn, seed: int | None = None) -> AgentEpisodeResult:
    observation, _ = env.reset(seed=seed)
    done = False
    mid_prices: list[float] = []
    rewards: list[float] = []
    pnls: list[float] = []
    inventories: list[float] = []
    bid_quotes: list[float] = []
    ask_quotes: list[float] = []
    spread_capture: list[float] = []
    trade_pnls: list[float] = []

    while not done:
        time_remaining = 1.0 - env.step_index / max(env.config.episode_length, 1)
        action = policy_fn(observation, time_remaining)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(float(reward))
        mid_prices.append(float(info["mid_price"]))
        pnls.append(float(info["mark_to_market"]))
        inventories.append(float(info["inventory"]))
        bid_quotes.append(float(info["bid_quote"]))
        ask_quotes.append(float(info["ask_quote"]))
        spread_capture.append(float(info["spread_capture"]))
        trade_pnls.append(float(reward))

    cumulative_pnl = np.cumsum(rewards).tolist()
    drawdowns = _compute_drawdowns(cumulative_pnl)
    inventory_variance = float(np.var(inventories)) if inventories else 0.0
    total_pnl = float(cumulative_pnl[-1]) if cumulative_pnl else 0.0
    sharpe_ratio = _compute_sharpe(rewards)
    average_spread_captured = float(np.mean(spread_capture)) if spread_capture else 0.0
    trade_win_rate = float(np.mean(np.asarray(trade_pnls) > 0.0)) if trade_pnls else 0.0

    return AgentEpisodeResult(
        mid_prices=mid_prices,
        rewards=rewards,
        pnls=pnls,
        inventories=inventories,
        bid_quotes=bid_quotes,
        ask_quotes=ask_quotes,
        spread_capture=spread_capture,
        trade_pnls=trade_pnls,
        drawdowns=drawdowns,
        inventory_variance=inventory_variance,
        total_pnl=total_pnl,
        sharpe_ratio=sharpe_ratio,
        average_spread_captured=average_spread_captured,
        trade_win_rate=trade_win_rate,
        cumulative_pnl=cumulative_pnl,
    )


def compare_agents(
    training_result,
    episodes: int = 20,
    seed: int = 7,
    output_dir: str | Path = "artifacts/comparison",
    baseline_config: AvellanedaStoikovConfig | None = None,
) -> ComparisonSummary:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    market_config = training_result.market_config if hasattr(training_result, "market_config") else MarketMakerConfig(seed=seed)

    dqn_env = MarketMakerEnvironment(market_config)
    baseline_env = MarketMakerEnvironment(market_config)
    dqn_agent = training_result.agent if hasattr(training_result, "agent") else DQNAgent(dqn_env.observation_size, dqn_env.action_grid, DQNConfig(), device="cpu")
    baseline_agent = AvellanedaStoikovAgent(market_config, baseline_config)

    dqn_runs: list[AgentEpisodeResult] = []
    baseline_runs: list[AgentEpisodeResult] = []

    for episode in range(episodes):
        dqn_runs.append(
            _run_policy_episode(
                "DQN",
                dqn_env,
                lambda observation, time_remaining: int(dqn_agent.select_action(observation, deterministic=True)[0]),
                seed=seed + episode,
            )
        )
        baseline_runs.append(
            _run_policy_episode(
                "Avellaneda-Stoikov",
                baseline_env,
                lambda observation, time_remaining: baseline_agent.select_action(observation, time_remaining)[0],
                seed=seed + episode,
            )
        )

    dqn_result = _aggregate_results(dqn_runs)
    baseline_result = _aggregate_results(baseline_runs)

    rows = [
        _summary_row("DQN", dqn_result),
        _summary_row("Avellaneda-Stoikov", baseline_result),
    ]

    if dqn_runs:
        plot_comparison_dashboard(dqn_runs[0], baseline_runs[0], training_result.history, output_path / "comparison_dashboard.png")
        _plot_distribution_comparison(dqn_runs, baseline_runs, output_path / "pnl_distribution.png")

    return ComparisonSummary(rows=rows, dqn_result=dqn_result, baseline_result=baseline_result, output_dir=output_path)


def _aggregate_results(results: list[AgentEpisodeResult]) -> AgentEpisodeResult:
    if not results:
        return AgentEpisodeResult([], [], [], [], [], [], [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, [])

    combined_mid_prices = [price for result in results for price in result.mid_prices]
    combined_rewards = [reward for result in results for reward in result.rewards]
    combined_pnls = [pnl for result in results for pnl in result.pnls]
    combined_inventories = [inventory for result in results for inventory in result.inventories]
    combined_bid_quotes = [quote for result in results for quote in result.bid_quotes]
    combined_ask_quotes = [quote for result in results for quote in result.ask_quotes]
    combined_spread_capture = [capture for result in results for capture in result.spread_capture]
    combined_trade_pnls = [trade for result in results for trade in result.trade_pnls]
    combined_drawdowns = [drawdown for result in results for drawdown in result.drawdowns]
    combined_cumulative_pnl = [value for result in results for value in result.cumulative_pnl]

    inventory_variance = float(np.mean([result.inventory_variance for result in results]))
    total_pnl = float(np.mean([result.total_pnl for result in results]))
    sharpe_ratio = float(np.mean([result.sharpe_ratio for result in results]))
    average_spread_captured = float(np.mean([result.average_spread_captured for result in results]))
    trade_win_rate = float(np.mean([result.trade_win_rate for result in results]))

    return AgentEpisodeResult(
        mid_prices=combined_mid_prices,
        rewards=combined_rewards,
        pnls=combined_pnls,
        inventories=combined_inventories,
        bid_quotes=combined_bid_quotes,
        ask_quotes=combined_ask_quotes,
        spread_capture=combined_spread_capture,
        trade_pnls=combined_trade_pnls,
        drawdowns=combined_drawdowns,
        inventory_variance=inventory_variance,
        total_pnl=total_pnl,
        sharpe_ratio=sharpe_ratio,
        average_spread_captured=average_spread_captured,
        trade_win_rate=trade_win_rate,
        cumulative_pnl=combined_cumulative_pnl,
    )


def _summary_row(name: str, result: AgentEpisodeResult) -> dict[str, float | str]:
    max_drawdown = float(min(result.drawdowns)) if result.drawdowns else 0.0
    return {
        "agent": name,
        "total_pnl": result.total_pnl,
        "sharpe_ratio": result.sharpe_ratio,
        "inventory_variance": result.inventory_variance,
        "average_spread_captured": result.average_spread_captured,
        "max_drawdown": max_drawdown,
        "trade_win_rate": result.trade_win_rate,
    }


def format_summary_table(rows: list[dict[str, float | str]]) -> str:
    headers = ["agent", "total_pnl", "sharpe_ratio", "inventory_variance", "average_spread_captured", "max_drawdown", "trade_win_rate"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(f"{row[header]:.4f}" if isinstance(row[header], (float, int)) else str(row[header])))

    def _format_cell(header: str, row: dict[str, float | str]) -> str:
        value = row[header]
        return f"{value:.4f}" if isinstance(value, (float, int)) else str(value)

    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    row_lines = [" | ".join(_format_cell(header, row).ljust(widths[header]) for header in headers) for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def _plot_distribution_comparison(dqn_runs: list[AgentEpisodeResult], baseline_runs: list[AgentEpisodeResult], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    dqn_pnls = [result.total_pnl for result in dqn_runs]
    baseline_pnls = [result.total_pnl for result in baseline_runs]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(dqn_pnls, bins=min(max(len(dqn_pnls) // 2, 10), 40), alpha=0.75, color="#0B6E99", label="DQN")
    axes[0].hist(baseline_pnls, bins=min(max(len(baseline_pnls) // 2, 10), 40), alpha=0.75, color="#E76F51", label="Avellaneda-Stoikov")
    axes[0].set_title("PnL Distribution Across Episodes")
    axes[0].set_xlabel("PnL")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].boxplot([dqn_pnls, baseline_pnls], labels=["DQN", "Avellaneda-Stoikov"], patch_artist=True, boxprops={"facecolor": "#D0E3F0"})
    axes[1].set_title("PnL Distribution Summary")
    axes[1].set_ylabel("PnL")
    axes[1].grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")


def plot_comparison_dashboard(dqn_result: AgentEpisodeResult, baseline_result: AgentEpisodeResult, history: dict[str, list[float]], output_path: str | Path | None = None):
    import matplotlib.pyplot as plt

    steps_dqn = np.arange(len(dqn_result.cumulative_pnl))
    steps_baseline = np.arange(len(baseline_result.cumulative_pnl))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(steps_dqn, dqn_result.cumulative_pnl, label="DQN", color="#0B6E99")
    axes[0, 0].plot(steps_baseline, baseline_result.cumulative_pnl, label="Avellaneda-Stoikov", color="#E76F51")
    axes[0, 0].set_title("Cumulative PnL Curves")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("PnL")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps_dqn, dqn_result.inventories, label="DQN", color="#7A5195")
    axes[0, 1].plot(steps_baseline, baseline_result.inventories, label="Avellaneda-Stoikov", color="#2A9D8F")
    axes[0, 1].set_title("Inventory Paths")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Inventory")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(True, alpha=0.3)

    dqn_mid = np.asarray(dqn_result.mid_prices, dtype=np.float32)
    baseline_mid = np.asarray(baseline_result.mid_prices, dtype=np.float32)
    dqn_bid_distance = dqn_mid - np.asarray(dqn_result.bid_quotes, dtype=np.float32)
    dqn_ask_distance = np.asarray(dqn_result.ask_quotes, dtype=np.float32) - dqn_mid
    baseline_bid_distance = baseline_mid - np.asarray(baseline_result.bid_quotes, dtype=np.float32)
    baseline_ask_distance = np.asarray(baseline_result.ask_quotes, dtype=np.float32) - baseline_mid
    axes[1, 0].plot(np.arange(len(dqn_bid_distance)), dqn_bid_distance, label="DQN bid distance", color="#0B6E99")
    axes[1, 0].plot(np.arange(len(dqn_ask_distance)), dqn_ask_distance, label="DQN ask distance", color="#4C78A8")
    axes[1, 0].plot(np.arange(len(baseline_bid_distance)), baseline_bid_distance, label="Baseline bid distance", color="#E76F51")
    axes[1, 0].plot(np.arange(len(baseline_ask_distance)), baseline_ask_distance, label="Baseline ask distance", color="#F4A261")
    axes[1, 0].set_title("Quote Distances From Mid Price")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Distance")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(True, alpha=0.3)

    episodes = history.get("episode", list(range(len(history.get("episode_reward", [])))))
    axes[1, 1].plot(episodes, history.get("episode_reward", []), label="Training reward", color="#D1495B")
    axes[1, 1].set_title("Training Reward Curve")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].grid(True, alpha=0.3)

    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    return figure
