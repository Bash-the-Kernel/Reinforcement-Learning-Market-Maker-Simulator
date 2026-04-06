"""DQN agent for market making quote selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import DQNConfig


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


@dataclass(slots=True)
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, observation_size: int):
        self.capacity = capacity
        self.observation_size = observation_size
        self.observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch:
        indices = rng.integers(0, self.size, size=batch_size)
        return TransitionBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
        )


class DQNAgent:
    def __init__(self, observation_size: int, action_grid: list[tuple[float, float]], config: DQNConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.action_grid = action_grid
        self.q_network = QNetwork(observation_size, len(action_grid), config.hidden_size).to(self.device)
        self.target_network = QNetwork(observation_size, len(action_grid), config.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.replay_capacity, observation_size)
        self.rng = np.random.default_rng()
        self.training_steps = 0

    def epsilon(self) -> float:
        progress = min(self.training_steps / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> tuple[int, np.ndarray, float]:
        epsilon = 0.0 if deterministic else self.epsilon()
        if not deterministic and self.rng.random() < epsilon:
            action_index = int(self.rng.integers(0, len(self.action_grid)))
        else:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(observation_tensor)
                action_index = int(torch.argmax(q_values, dim=-1).item())

        action = np.asarray(self.action_grid[action_index], dtype=np.float32)
        return action_index, action, epsilon

    def add_transition(self, observation: np.ndarray, action_index: int, reward: float, next_observation: np.ndarray, done: bool) -> None:
        self.replay_buffer.add(observation, action_index, reward, next_observation, done)

    def update(self) -> dict[str, float] | None:
        if self.replay_buffer.size < max(self.config.batch_size, self.config.warmup_steps):
            self.training_steps += 1
            return None

        batch = self.replay_buffer.sample(self.config.batch_size, self.rng)
        observations = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(batch.next_observations, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        q_values = self.q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_observations).max(dim=1).values
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q_values

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(loss.item()),
            "epsilon": self.epsilon(),
        }

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())
