"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
import torch.nn.functional as F


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim * chunk_size))
        self.net = nn.Sequential(*layers)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        chunks_pred = self.sample_actions(state)
        return F.mse_loss(chunks_pred, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        chunks_raw = self.net(state)
        chunks = chunks_raw.view(-1, self.chunk_size, self.action_dim)
        return chunks


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        prev_dim = state_dim + (action_dim * chunk_size) + 1
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim * chunk_size))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        device = state.device

        noise = torch.randn_like(action_chunk)
        tau = torch.rand(batch_size, 1,1, device=device)
        # interpolation
        noisy_action = tau * action_chunk + (1-tau) * noise
        # target velocity
        target_velocity = action_chunk - noise
        # flatten and concatenate
        noisy_action_flat = noisy_action.view(batch_size, -1)
        tau_flat = tau.view(batch_size,1)
        net_input = torch.cat([state, noisy_action_flat, tau_flat], dim=-1)
        # forward pass
        predicted_velocity_flat = self.net(net_input)
        predicted_velocity = predicted_velocity_flat.view(batch_size, self.chunk_size, self.action_dim)

        return F.mse_loss(predicted_velocity, target_velocity)
        

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        device = state.device

        action = torch.randn(batch_size, self.chunk_size, self.action_dim, device=device)
        dt = 1/num_steps
        for i in range(num_steps):
            tau = i * dt

            action_flat = action.view(batch_size, -1)
            tau_tensor = torch.full((batch_size, 1), tau, device=device)
            net_input = torch.cat([state, action_flat, tau_tensor], dim=-1)

            predicted_velocity_flat = self.net(net_input)
            predicted_velocity = predicted_velocity_flat.view(batch_size, self.chunk_size, self.action_dim)
            action = action + predicted_velocity * dt
        
        return action
            



PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
