"""Simplified limit order book market making environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import MarketMakerConfig


@dataclass(slots=True)
class StepInfo:
    mid_price: float
    market_spread_bps: float
    best_bid: float
    best_ask: float
    bid_quote: float
    ask_quote: float
    inventory: float
    inventory_change: float
    cash: float
    mark_to_market: float
    spread_capture: float
    executed_bid_volume: int
    executed_ask_volume: int
    bid_fills: int
    ask_fills: int


@dataclass(slots=True)
class BoxSpace:
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or np.random.default_rng()
        return generator.uniform(self.low, self.high).astype(self.dtype, copy=False)

    def contains(self, value: np.ndarray) -> bool:
        array = np.asarray(value, dtype=self.dtype)
        return array.shape == self.shape and np.all(array >= self.low) and np.all(array <= self.high)


@dataclass(slots=True)
class DiscreteSpace:
    n: int

    def sample(self, rng: np.random.Generator | None = None) -> int:
        generator = rng or np.random.default_rng()
        return int(generator.integers(0, self.n))

    def contains(self, value: int) -> bool:
        return isinstance(value, (int, np.integer)) and 0 <= int(value) < self.n


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
        self.best_bid = self.config.initial_price * (1.0 - self.config.base_spread_bps / 20_000.0)
        self.best_ask = self.config.initial_price * (1.0 + self.config.base_spread_bps / 20_000.0)
        self.recent_return = 0.0
        self.short_term_momentum = 0.0
        self._prev_mid_price = self.config.initial_price
        self.action_grid = [(float(bid), float(ask)) for bid in self.config.quote_offset_levels for ask in self.config.quote_offset_levels]
        self.observation_space = BoxSpace(
            low=np.array([0.0, -self.config.inventory_limit, 0.0, -1.0], dtype=np.float32),
            high=np.array([np.finfo(np.float32).max, self.config.inventory_limit, np.finfo(np.float32).max, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = DiscreteSpace(len(self.action_grid))

    @property
    def observation_size(self) -> int:
        return int(self.observation_space.shape[0])

    @property
    def action_size(self) -> int:
        return int(self.action_space.shape[0])

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_index = 0
        self.mid_price = self.config.initial_price
        self._prev_mid_price = self.mid_price
        self.inventory = 0.0
        self.cash = 0.0
        self.market_spread_bps = self.config.base_spread_bps
        self.best_bid = self.mid_price * (1.0 - self.market_spread_bps / 20_000.0)
        self.best_ask = self.mid_price * (1.0 + self.market_spread_bps / 20_000.0)
        self.recent_return = 0.0
        self.short_term_momentum = 0.0
        return self._get_observation(), self._build_info(0, 0, 0.0, 0.0, 0.0, 0, 0, 0.0)

    def step(self, action: int | np.integer | np.ndarray | list[float] | tuple[float, float]) -> tuple[np.ndarray, float, bool, bool, dict]:
        bid_offset_bps, ask_offset_bps = self._action_to_offsets(action)

        pre_step_mid = self.mid_price
        old_wealth = self.cash + self.inventory * pre_step_mid

        self._evolve_mid_price()
        self._update_market_spread()

        bid_quote = self.mid_price * (1.0 - bid_offset_bps / 10_000.0)
        ask_quote = self.mid_price * (1.0 + ask_offset_bps / 10_000.0)

        market_buy_arrivals = int(self.rng.poisson(max(self.config.order_intensity, 0.0)))
        market_sell_arrivals = int(self.rng.poisson(max(self.config.order_intensity, 0.0)))

        competitive_ask = ask_quote <= self.best_ask
        competitive_bid = bid_quote >= self.best_bid

        ask_fills = int(min(market_buy_arrivals if competitive_ask else 0, self.config.max_fill_per_step))
        bid_fills = int(min(market_sell_arrivals if competitive_bid else 0, self.config.max_fill_per_step))

        executed_ask_volume = ask_fills
        executed_bid_volume = bid_fills
        inventory_change = float(executed_bid_volume - executed_ask_volume)

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

        info = self._build_info(
            bid_fills,
            ask_fills,
            bid_quote,
            ask_quote,
            spread_capture,
            executed_bid_volume,
            executed_ask_volume,
            inventory_change,
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _action_to_offsets(self, action: int | np.integer | np.ndarray | list[float] | tuple[float, float]) -> tuple[float, float]:
        if isinstance(action, (int, np.integer)):
            bid_offset, ask_offset = self.action_grid[int(action)]
            return bid_offset, ask_offset

        array = np.asarray(action, dtype=np.float64)
        if array.shape != (2,):
            raise ValueError("Action must be a discrete index or a length-2 bid/ask offset pair.")

        if np.all((-1.0 <= array) & (array <= 1.0)):
            span = self.config.max_quote_bps - self.config.min_quote_bps
            bid_offset = self.config.min_quote_bps + (array[0] + 1.0) * 0.5 * span
            ask_offset = self.config.min_quote_bps + (array[1] + 1.0) * 0.5 * span
            return float(bid_offset), float(ask_offset)

        return float(array[0]), float(array[1])

    def _evolve_mid_price(self) -> None:
        drift = (self.config.annual_drift - 0.5 * self.config.annual_volatility ** 2) * self.config.dt
        shock = self.config.annual_volatility * np.sqrt(self.config.dt) * self.rng.normal()
        prev_mid = self.mid_price
        self.mid_price = float(prev_mid * np.exp(drift + shock))
        self.recent_return = float(np.log(self.mid_price / max(prev_mid, 1e-12)))
        self.short_term_momentum = float(0.8 * self.short_term_momentum + 0.2 * self.recent_return)
        self._prev_mid_price = prev_mid

    def _update_market_spread(self) -> None:
        realized_vol = abs(self.recent_return) / max(np.sqrt(self.config.dt), 1e-12)
        target_spread = self.config.base_spread_bps * (1.0 + self.config.spread_volatility_scale * realized_vol)
        self.market_spread_bps = float(np.clip(0.85 * self.market_spread_bps + 0.15 * target_spread, 1.0, 5.0 * self.config.base_spread_bps))
        self.best_bid = self.mid_price * (1.0 - self.market_spread_bps / 20_000.0)
        self.best_ask = self.mid_price * (1.0 + self.market_spread_bps / 20_000.0)

    def _get_observation(self) -> np.ndarray:
        obs = np.array(
            [
                self.mid_price,
                self.inventory,
                self.market_spread_bps,
                self.short_term_momentum,
            ],
            dtype=np.float32,
        )
        return obs

    def _build_info(
        self,
        bid_fills: int,
        ask_fills: int,
        bid_quote: float,
        ask_quote: float,
        spread_capture: float,
        executed_bid_volume: int,
        executed_ask_volume: int,
        inventory_change: float,
    ) -> dict:
        mark_to_market = self.cash + self.inventory * self.mid_price
        info = StepInfo(
            mid_price=self.mid_price,
            market_spread_bps=self.market_spread_bps,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            bid_quote=bid_quote,
            ask_quote=ask_quote,
            inventory=self.inventory,
            inventory_change=inventory_change,
            cash=self.cash,
            mark_to_market=mark_to_market,
            spread_capture=spread_capture,
            executed_bid_volume=executed_bid_volume,
            executed_ask_volume=executed_ask_volume,
            bid_fills=bid_fills,
            ask_fills=ask_fills,
        )
        return asdict(info)
