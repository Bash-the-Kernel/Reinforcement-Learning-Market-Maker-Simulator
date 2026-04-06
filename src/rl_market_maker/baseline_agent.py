"""Avellaneda-Stoikov baseline quoting strategy."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np

from .config import AvellanedaStoikovConfig, MarketMakerConfig


@dataclass(slots=True)
class BaselineQuote:
    bid_offset_bps: float
    ask_offset_bps: float
    reservation_price: float
    optimal_spread: float
    volatility: float
    time_remaining: float


class AvellanedaStoikovAgent:
    """Theoretical market making baseline based on Avellaneda-Stoikov."""

    def __init__(self, market_config: MarketMakerConfig, config: AvellanedaStoikovConfig | None = None):
        self.market_config = market_config
        self.config = config or AvellanedaStoikovConfig(inventory_limit=market_config.inventory_limit, min_quote_bps=market_config.min_quote_bps, max_quote_bps=market_config.max_quote_bps)
        self._returns = deque(maxlen=max(2, self.config.volatility_window))

    def observe(self, state: np.ndarray) -> None:
        mid_price = float(state[0])
        momentum = float(state[3])
        self._returns.append(momentum)
        if mid_price <= 0.0:
            return

    def select_action(self, observation: np.ndarray, time_remaining: float) -> tuple[np.ndarray, BaselineQuote]:
        mid_price = float(observation[0])
        inventory = float(observation[1])
        spread_bps = max(float(observation[2]), self.market_config.min_quote_bps)
        momentum = float(observation[3])
        self._returns.append(momentum)
        sigma = self._estimate_volatility(mid_price, momentum, spread_bps)

        gamma = self.config.gamma
        k = max(self.config.k, 1e-6)
        reservation_price = mid_price - inventory * gamma * sigma**2 * time_remaining
        optimal_spread = gamma * sigma**2 * time_remaining + (2.0 / gamma) * np.log(1.0 + gamma / k)

        bid_quote = reservation_price - optimal_spread / 2.0
        ask_quote = reservation_price + optimal_spread / 2.0

        if self.config.quote_clip_to_market:
            bid_quote = min(max(bid_quote, mid_price * (1.0 - self.config.max_quote_bps / 10_000.0)), mid_price * (1.0 - self.config.min_quote_bps / 10_000.0))
            ask_quote = max(min(ask_quote, mid_price * (1.0 + self.config.max_quote_bps / 10_000.0)), mid_price * (1.0 + self.config.min_quote_bps / 10_000.0))

        bid_offset_bps = max((mid_price - bid_quote) / max(mid_price, 1e-12) * 10_000.0, self.config.min_quote_bps)
        ask_offset_bps = max((ask_quote - mid_price) / max(mid_price, 1e-12) * 10_000.0, self.config.min_quote_bps)

        quote = BaselineQuote(
            bid_offset_bps=float(np.clip(bid_offset_bps, self.config.min_quote_bps, self.config.max_quote_bps)),
            ask_offset_bps=float(np.clip(ask_offset_bps, self.config.min_quote_bps, self.config.max_quote_bps)),
            reservation_price=float(reservation_price),
            optimal_spread=float(optimal_spread),
            volatility=float(sigma),
            time_remaining=float(time_remaining),
        )
        return np.asarray([quote.bid_offset_bps, quote.ask_offset_bps], dtype=np.float32), quote

    def _estimate_volatility(self, mid_price: float, momentum: float, spread_bps: float) -> float:
        rolling_component = float(np.std(self._returns)) * mid_price if len(self._returns) > 1 else 0.0
        momentum_component = abs(momentum) * mid_price
        spread_component = spread_bps / 10_000.0 * mid_price
        base_vol = self.market_config.annual_volatility * np.sqrt(self.market_config.dt) * mid_price
        return float(max(base_vol, 0.5 * rolling_component + 0.3 * momentum_component + 0.2 * spread_component, 1e-6))
