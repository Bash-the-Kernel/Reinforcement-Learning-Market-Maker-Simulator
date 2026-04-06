"""Configuration objects for the simulator and PPO trainer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MarketMakerConfig:
    episode_length: int = 512
    initial_price: float = 100.0
    dt: float = 1.0 / 252.0
    annual_drift: float = 0.0
    annual_volatility: float = 0.18
    base_spread_bps: float = 8.0
    spread_volatility_scale: float = 3.0
    order_intensity: float = 5.0
    fill_sensitivity: float = 1.7
    min_quote_bps: float = 1.0
    max_quote_bps: float = 40.0
    inventory_limit: float = 20.0
    inventory_penalty: float = 0.0015
    transaction_cost_bps: float = 0.35
    max_fill_per_step: int = 3
    seed: int | None = None


@dataclass(slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    update_epochs: int = 8
    batch_size: int = 64
    rollout_size: int = 2048
    hidden_size: int = 128
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    action_std_floor: float = 0.08


@dataclass(slots=True)
class TrainingConfig:
    episodes: int = 250
    evaluation_episodes: int = 5
    seed: int = 7
    output_dir: str = "artifacts"
    device: str = "cpu"
    save_plots: bool = True
    market: MarketMakerConfig = field(default_factory=MarketMakerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
