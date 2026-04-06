"""PPO agent for continuous market making actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .config import PPOConfig


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_mean = nn.Linear(hidden_size, action_size)
        self.policy_log_std = nn.Parameter(torch.full((action_size,), -0.5))
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.shared(observation)
        mean = self.policy_mean(latent)
        return mean, self.policy_log_std.expand_as(mean), self.value_head(latent).squeeze(-1)


@dataclass(slots=True)
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


class RolloutBuffer:
    def __init__(self):
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[float] = []
        self.returns: np.ndarray | None = None
        self.advantages: np.ndarray | None = None

    def add(self, observation: np.ndarray, action: np.ndarray, log_prob: float, reward: float, done: bool, value: float) -> None:
        self.observations.append(observation.astype(np.float32, copy=False))
        self.actions.append(action.astype(np.float32, copy=False))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[index]
            delta = rewards[index] + gamma * values[index + 1] * non_terminal - values[index]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[index] = gae

        self.advantages = advantages
        self.returns = advantages + np.asarray(self.values, dtype=np.float32)

    def as_batch(self) -> RolloutBatch:
        if self.returns is None or self.advantages is None:
            raise RuntimeError("Call compute_returns_and_advantages before requesting a batch.")

        return RolloutBatch(
            observations=np.asarray(self.observations, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            log_probs=np.asarray(self.log_probs, dtype=np.float32),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            dones=np.asarray(self.dones, dtype=np.float32),
            values=np.asarray(self.values, dtype=np.float32),
            returns=self.returns.astype(np.float32, copy=False),
            advantages=self.advantages.astype(np.float32, copy=False),
        )

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.returns = None
        self.advantages = None


class PPOAgent:
    def __init__(self, observation_size: int, action_size: int, config: PPOConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = ActorCritic(observation_size, action_size, config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.action_size = action_size

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std, value = self.model(observation_tensor)
            std = torch.exp(log_std).clamp_min(self.config.action_std_floor)
            distribution = Normal(mean, std)
            raw_action = mean if deterministic else distribution.rsample()
            action = torch.tanh(raw_action)
            log_prob = distribution.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)

        return action.squeeze(0).cpu().numpy(), float(log_prob.sum(dim=-1).item()), float(value.item())

    def evaluate_observations(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, values = self.model(observations)
        std = torch.exp(log_std).clamp_min(self.config.action_std_floor)
        distribution = Normal(mean, std)
        raw_actions = _atanh(actions)
        log_prob = distribution.log_prob(raw_actions) - torch.log(1.0 - actions.pow(2) + 1e-6)
        entropy = distribution.entropy()
        return log_prob.sum(dim=-1), entropy.sum(dim=-1), values

    def update(self, rollout: RolloutBatch) -> dict[str, float]:
        observations = torch.as_tensor(rollout.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout.actions, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(rollout.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(rollout.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(rollout.advantages, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        sample_count = observations.shape[0]
        batch_size = min(self.config.batch_size, sample_count)

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        update_count = 0

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_observations = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                log_probs, entropy, values = self.evaluate_observations(batch_observations, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
                policy_loss = -(torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages)).mean()
                value_loss = 0.5 * (batch_returns - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy_loss.item())
                update_count += 1

        return {
            "policy_loss": policy_loss_total / max(update_count, 1),
            "value_loss": value_loss_total / max(update_count, 1),
            "entropy": entropy_total / max(update_count, 1),
        }
