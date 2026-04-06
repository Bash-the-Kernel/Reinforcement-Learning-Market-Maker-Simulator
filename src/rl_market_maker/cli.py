"""Command line entry point for training the simulator."""

from __future__ import annotations

import argparse
from pprint import pprint

from .comparison import compare_agents, format_summary_table
from .config import DQNConfig, MarketMakerConfig, TrainingConfig
from .training import train_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or compare market making agents.")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--no-save-plots", action="store_true")
    parser.add_argument("--compare-agents", action="store_true", help="Train DQN then compare against the Avellaneda-Stoikov baseline.")
    parser.add_argument("--comparison-episodes", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    save_plots = True
    if args.no_save_plots:
        save_plots = False
    elif args.save_plots:
        save_plots = True

    config = TrainingConfig(
        episodes=args.episodes,
        evaluation_episodes=args.evaluation_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        save_plots=save_plots,
        market=MarketMakerConfig(seed=args.seed),
        dqn=DQNConfig(),
    )
    result = train_agent(config)

    if args.compare_agents:
        comparison = compare_agents(
            result,
            episodes=args.comparison_episodes,
            seed=args.seed,
            output_dir=f"{args.output_dir}/comparison",
        )
        print(format_summary_table(comparison.rows))
    else:
        pprint({"final_metrics": result.evaluation[-1] if result.evaluation else {}, "training_episodes": len(result.history["episode"])})


if __name__ == "__main__":
    main()
