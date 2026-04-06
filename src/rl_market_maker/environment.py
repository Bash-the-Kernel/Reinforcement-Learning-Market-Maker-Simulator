"""Simplified limit order book market making environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import MarketMakerConfig


@dataclass(slots=True)
class StepInfo:
    mid_price: float
    market_spread_bps: float
    bid_quote: float
    ask_quote: float
    inventory: float
    cash: float
    mark_to_market: float
    spread_capture: float
    bid_fills: int
    ask_fills: int


class MarketMakerEnvironment:
    """Market maker simulator with GBM mid-price dynamics and Poisson fills."""

    def __init__(self, config: MarketMakerConfig | None = None):
        self.config = config or MarketMakerConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.step_index = 0
        self.mid_price = self.config.initial_price
        self.inventory = 0.0
        self.cash = 0.0
        self.market_spread_bps = self.config.base_spread_bps
        self.recent_return = 0.0
        self._prev_mid_price = self.config.initial_price

    @property
    def observation_size(self) -> int:
        return 5

    @property
    def action_size(self) -> int:
        return 2

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_index = 0
        self.mid_price = self.config.initial_price
        self._prev_mid_price = self.mid_price
        self.inventory = 0.0
        self.cash = 0.0
        self.market_spread_bps = self.config.base_spread_bps
        self.recent_return = 0.0
        return self._get_observation(), self._build_info(0, 0, 0.0, 0.0, 0.0)

    def step(self, action: np.ndarray | list[float] | tuple[float, float]) -> tuple[np.ndarray, float, bool, bool, dict]:
        bid_offset_bps, ask_offset_bps = self._map_action_to_offsets(np.asarray(action, dtype=np.float64))

        pre_step_mid = self.mid_price
        old_wealth = self.cash + self.inventory * pre_step_mid

        self._evolve_mid_price()
        self._update_market_spread()

        bid_quote = self.mid_price * (1.0 - bid_offset_bps / 10_000.0)
        ask_quote = self.mid_price * (1.0 + ask_offset_bps / 10_000.0)
        market_bid = self.mid_price * (1.0 - self.market_spread_bps / 20_000.0)
        market_ask = self.mid_price * (1.0 + self.market_spread_bps / 20_000.0)

        bid_gap_bps = max((market_bid - bid_quote) / self.mid_price * 10_000.0, 0.0)
        ask_gap_bps = max((ask_quote - market_ask) / self.mid_price * 10_000.0, 0.0)

        bid_lambda = self.config.order_intensity * np.exp(-self.config.fill_sensitivity * bid_gap_bps / max(self.market_spread_bps, 1e-6))
        ask_lambda = self.config.order_intensity * np.exp(-self.config.fill_sensitivity * ask_gap_bps / max(self.market_spread_bps, 1e-6))

        bid_fills = int(min(self.rng.poisson(max(bid_lambda, 0.0)), self.config.max_fill_per_step))
        ask_fills = int(min(self.rng.poisson(max(ask_lambda, 0.0)), self.config.max_fill_per_step))

        execution_cost = 0.0
        if bid_fills > 0:
            self.inventory += bid_fills
            self.cash -= bid_quote * bid_fills
            execution_cost += bid_fills * bid_quote * self.config.transaction_cost_bps / 10_000.0
        if ask_fills > 0:
            self.inventory -= ask_fills
            self.cash += ask_quote * ask_fills
            execution_cost += ask_fills * ask_quote * self.config.transaction_cost_bps / 10_000.0

        new_wealth = self.cash + self.inventory * self.mid_price
        realized_pnl = new_wealth - old_wealth
        inventory_penalty = self.config.inventory_penalty * (self.inventory / max(self.config.inventory_limit, 1e-6)) ** 2 * self.mid_price
        reward = realized_pnl - inventory_penalty - execution_cost

        spread_capture = 0.0
        if bid_fills > 0:
            spread_capture += (self.mid_price - bid_quote) * bid_fills
        if ask_fills > 0:
            spread_capture += (ask_quote - self.mid_price) * ask_fills

        self.step_index += 1
        terminated = self.step_index >= self.config.episode_length
        truncated = False

        info = self._build_info(bid_fills, ask_fills, bid_quote, ask_quote, spread_capture)
        return self._get_observation(), float(reward), terminated, truncated, info

    def _map_action_to_offsets(self, action: np.ndarray) -> tuple[float, float]:
        bounded = np.clip(action, -1.0, 1.0)
        span = self.config.max_quote_bps - self.config.min_quote_bps
        bid_offset = self.config.min_quote_bps + (bounded[0] + 1.0) * 0.5 * span
        ask_offset = self.config.min_quote_bps + (bounded[1] + 1.0) * 0.5 * span
        return float(bid_offset), float(ask_offset)

    def _evolve_mid_price(self) -> None:
        drift = (self.config.annual_drift - 0.5 * self.config.annual_volatility ** 2) * self.config.dt
        shock = self.config.annual_volatility * np.sqrt(self.config.dt) * self.rng.normal()
        prev_mid = self.mid_price
        self.mid_price = float(prev_mid * np.exp(drift + shock))
        self.recent_return = float(np.log(self.mid_price / max(prev_mid, 1e-12)))
        self._prev_mid_price = prev_mid

    def _update_market_spread(self) -> None:
        realized_vol = abs(self.recent_return) / max(np.sqrt(self.config.dt), 1e-12)
        target_spread = self.config.base_spread_bps * (1.0 + self.config.spread_volatility_scale * realized_vol)
        self.market_spread_bps = float(np.clip(0.85 * self.market_spread_bps + 0.15 * target_spread, 1.0, 5.0 * self.config.base_spread_bps))

    def _get_observation(self) -> np.ndarray:
        time_remaining = 1.0 - self.step_index / max(self.config.episode_length, 1)
        obs = np.array(
            [
                self.mid_price,
                self.inventory,
                self.market_spread_bps,
                self.recent_return,
                time_remaining,
            ],
            dtype=np.float32,
        )
        return obs

    def _build_info(self, bid_fills: int, ask_fills: int, bid_quote: float, ask_quote: float, spread_capture: float) -> dict:
        mark_to_market = self.cash + self.inventory * self.mid_price
        info = StepInfo(
            mid_price=self.mid_price,
            market_spread_bps=self.market_spread_bps,
            bid_quote=bid_quote,
            ask_quote=ask_quote,
            inventory=self.inventory,
            cash=self.cash,
            mark_to_market=mark_to_market,
            spread_capture=spread_capture,
            bid_fills=bid_fills,
            ask_fills=ask_fills,
        )
        return asdict(info)
