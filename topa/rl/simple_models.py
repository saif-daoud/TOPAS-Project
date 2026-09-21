import torch
from torch import nn


class MLPPolicy(nn.Module):
    """Small discrete policy/Q head for latent states and optional conv-state features."""

    def __init__(
        self,
        in_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        conv_state_dim: int = 0,
        projection_dim: int = 0,
        conv_state_embedding_dim: int = 0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.conv_state_dim = int(conv_state_dim)
        self.latent_dim = self.in_dim - self.conv_state_dim
        self.projection_dim = int(projection_dim) if int(projection_dim) > 0 else (self.hidden_dim if self.hidden_dim > 0 else 0)
        self.conv_state_embedding_dim = int(conv_state_embedding_dim) if int(conv_state_embedding_dim) > 0 else 0

        if self.conv_state_dim > 0 and self.projection_dim > 0:
            self.latent_proj = nn.Sequential(
                nn.Linear(self.latent_dim, self.projection_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )
        else:
            self.latent_proj = None

        if self.conv_state_dim > 0 and self.conv_state_embedding_dim > 0:
            self.conv_state_proj = nn.Sequential(
                nn.Linear(self.conv_state_dim, self.conv_state_embedding_dim),
            )
        else:
            self.conv_state_proj = None

        if self.conv_state_dim > 0:
            latent_out_dim = self.projection_dim if self.latent_proj is not None else self.latent_dim
            conv_out_dim = self.conv_state_embedding_dim if self.conv_state_proj is not None else self.conv_state_dim
            net_in_dim = latent_out_dim + conv_out_dim
        else:
            net_in_dim = self.in_dim

        if self.hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(int(net_in_dim), self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, int(num_actions)),
            )
        else:
            self.net = nn.Linear(int(net_in_dim), int(num_actions))

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        if self.conv_state_dim > 0:
            z = input_embeds[:, : self.latent_dim]
            c = input_embeds[:, self.latent_dim :]
            if self.latent_proj is not None:
                z = self.latent_proj(z)
            if self.conv_state_proj is not None:
                c = self.conv_state_proj(c)
            input_embeds = torch.cat([z, c], dim=-1)
        return self.net(input_embeds)


class MLPValue(nn.Module):
    """Small scalar value head for activation-vector states."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        conv_state_dim: int = 0,
        projection_dim: int = 0,
        conv_state_embedding_dim: int = 0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.conv_state_dim = int(conv_state_dim)
        self.latent_dim = self.in_dim - self.conv_state_dim
        self.projection_dim = int(projection_dim) if int(projection_dim) > 0 else (self.hidden_dim if self.hidden_dim > 0 else 0)
        self.conv_state_embedding_dim = int(conv_state_embedding_dim) if int(conv_state_embedding_dim) > 0 else 0

        if self.conv_state_dim > 0 and self.projection_dim > 0:
            self.latent_proj = nn.Sequential(
                nn.Linear(self.latent_dim, self.projection_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )
        else:
            self.latent_proj = None

        if self.conv_state_dim > 0 and self.conv_state_embedding_dim > 0:
            self.conv_state_proj = nn.Sequential(
                nn.Linear(self.conv_state_dim, self.conv_state_embedding_dim),
            )
        else:
            self.conv_state_proj = None

        if self.conv_state_dim > 0:
            latent_out_dim = self.projection_dim if self.latent_proj is not None else self.latent_dim
            conv_out_dim = self.conv_state_embedding_dim if self.conv_state_proj is not None else self.conv_state_dim
            net_in_dim = latent_out_dim + conv_out_dim
        else:
            net_in_dim = self.in_dim

        if self.hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(int(net_in_dim), self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(int(net_in_dim), 1)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        if self.conv_state_dim > 0:
            z = input_embeds[:, : self.latent_dim]
            c = input_embeds[:, self.latent_dim :]
            if self.latent_proj is not None:
                z = self.latent_proj(z)
            if self.conv_state_proj is not None:
                c = self.conv_state_proj(c)
            input_embeds = torch.cat([z, c], dim=-1)
        return self.net(input_embeds).squeeze(-1)
