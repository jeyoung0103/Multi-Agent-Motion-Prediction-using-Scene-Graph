"""
Graphormer components for QCNet integration with unified agent + map nodes.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import wrap_angle


def init_params(module, n_layers):
    """Initialize parameters for Graphormer components"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(max(n_layers, 1)))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """
    Build node embeddings for both agent and map nodes with dedicated centrality encodings.
    """

    def __init__(self,
                hidden_dim: int,
                num_agent_types: int,
                n_layers: int,
                num_node_types: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_agent_types = num_agent_types

        self.node_type_encoder = nn.Embedding(num_node_types, hidden_dim)
        self.agent_type_encoder = nn.Embedding(num_agent_types + 1, hidden_dim, padding_idx=0)

        self.agent_risk_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.map_flow_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self,
                agent_features: torch.Tensor,
                map_features: torch.Tensor,
                agent_types: torch.Tensor,
                agent_risk_scores: torch.Tensor,
                map_follow_scores: torch.Tensor) -> torch.Tensor:
        device = agent_features.device if agent_features.numel() > 0 else map_features.device
        node_embeddings = []

        if agent_features.numel() > 0:
            num_agents = agent_features.size(0)
            agent_embed = agent_features
            node_type = torch.zeros(num_agents, dtype=torch.long, device=device)
            agent_embed = agent_embed + self.node_type_encoder(node_type)
            agent_embed = agent_embed + self.agent_type_encoder(agent_types)
            agent_embed = agent_embed + self.agent_risk_encoder(agent_risk_scores.unsqueeze(-1))
            node_embeddings.append(agent_embed)

        if map_features.numel() > 0:
            num_maps = map_features.size(0)
            map_embed = map_features
            node_type = torch.ones(num_maps, dtype=torch.long, device=device)
            map_embed = map_embed + self.node_type_encoder(node_type)
            map_embed = map_embed + self.map_flow_encoder(map_follow_scores.unsqueeze(-1))
            node_embeddings.append(map_embed)

        if len(node_embeddings) == 0:
            node_feature = torch.empty(0, self.hidden_dim, device=device)
        else:
            node_feature = torch.cat(node_embeddings, dim=0)

        graph_token_feature = self.graph_token.weight.squeeze(0)
        graph_token_feature = graph_token_feature.unsqueeze(0)

        return torch.cat([graph_token_feature, node_feature], dim=0)


class GraphAttnBias(nn.Module):
    """
    Compute attention bias using spatial encoding and QCNet-style RPE for all nodes.
    """

    def __init__(self,
                 num_heads: int,
                 num_spatial_bins: int,
                 hidden_dim: int = 128,
                 n_layers: int = 6,
                 max_distance: float = 300.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_spatial_bins = num_spatial_bins
        self.hidden_dim = hidden_dim
        self.max_distance = max_distance

        self.spatial_pos_encoder = nn.Embedding(num_spatial_bins, num_heads, padding_idx=0)
        from layers.fourier_embedding import FourierEmbedding
        self.rpe_encoder = FourierEmbedding(input_dim=3, hidden_dim=hidden_dim, num_freq_bands=8)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self,
                positions: torch.Tensor,
                heading_angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [num_nodes, 2]
            heading_angles: [num_nodes]
        Returns:
            attn_bias: [num_nodes + 1, num_nodes + 1, num_heads]
        """
        num_nodes = positions.size(0)
        device = positions.device

        attn_bias = torch.zeros(num_nodes + 1, num_nodes + 1, self.num_heads, device=device)

        if num_nodes == 0:
            bias = self.graph_token_virtual_distance.weight.view(1, 1, self.num_heads)
            attn_bias[0, :, :] = bias
            attn_bias[:, 0, :] = bias
            return attn_bias

        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        x_diff = rel_pos[:, :, 0]
        y_diff = rel_pos[:, :, 1]

        bin_size = self.max_distance / self.num_spatial_bins
        x_bins = torch.clamp(((x_diff / bin_size) + self.num_spatial_bins // 2).long(),
                             0, self.num_spatial_bins - 1)
        y_bins = torch.clamp(((y_diff / bin_size) + self.num_spatial_bins // 2).long(),
                             0, self.num_spatial_bins - 1)

        spatial_bias = self.spatial_pos_encoder(x_bins) + self.spatial_pos_encoder(y_bins)
        attn_bias[1:, 1:, :] += spatial_bias

        distances = torch.norm(rel_pos, p=2, dim=-1)
        angles = torch.atan2(rel_pos[:, :, 1], rel_pos[:, :, 0])
        rel_heading = wrap_angle(heading_angles.unsqueeze(1) - heading_angles.unsqueeze(0))

        rpe_features = torch.stack([distances, angles, rel_heading], dim=-1).view(-1, 3)
        rpe_emb = self.rpe_encoder(continuous_inputs=rpe_features.float(), categorical_embs=None)
        rpe_emb = rpe_emb.view(num_nodes, num_nodes, self.hidden_dim)
        rpe_bias = rpe_emb[:, :, :self.num_heads]
        attn_bias[1:, 1:, :] += rpe_bias

        graph_token_bias = self.graph_token_virtual_distance.weight.view(1, 1, self.num_heads)
        attn_bias[1:, 0, :] = graph_token_bias
        attn_bias[0, :, :] = graph_token_bias

        return attn_bias


class GraphormerGraphEncoderLayer(nn.Module):
    """
    Graphormer layer with additive attention bias support.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float = 0.1,
                 activation_fn: str = "gelu",
                 pre_layernorm: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pre_layernorm = pre_layernorm

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Analysis helper: set True to store per-layer attention weights
        self.record_attn = False
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = getattr(F, activation_fn)

    def forward(self,
                x: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_mask = None
        if attn_bias is not None:
            attn_mask = attn_bias.mean(dim=-1)
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.squeeze(0)

        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        x, attn_weights = self.self_attn(
            x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False
        )
        if self.record_attn:
            # attn_weights: [batch, num_heads, tgt, src]
            self._last_attn = attn_weights.detach().cpu()
        x = self.dropout(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)

        return x


class GraphormerGraphEncoder(nn.Module):
    """
    Graphormer encoder that jointly models agent and map nodes.
    """

    def __init__(self,
                 num_heads: int,
                 num_agent_types: int,
                 num_speed_bins: int,  # retained for backwards compatibility
                 num_spatial_bins: int,
                 num_ttc_bins: int,
                 num_edge_types: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 agent_risk_temperature: float = 3.0,
                 map_flow_normalizer: float = 6.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.agent_risk_temperature = agent_risk_temperature
        self.map_flow_normalizer = map_flow_normalizer

        self.node_feature = GraphNodeFeature(
            hidden_dim=hidden_dim,
            num_agent_types=num_agent_types,
            n_layers=num_layers,
        )

        self.attn_bias = GraphAttnBias(
            num_heads=num_heads,
            num_spatial_bins=num_spatial_bins,
            hidden_dim=hidden_dim,
            n_layers=num_layers,
        )

        self.layers = nn.ModuleList([
            GraphormerGraphEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=hidden_dim // num_heads,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def compute_ttc(positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_vel = velocities.unsqueeze(1) - velocities.unsqueeze(0)
        rel_pos_dot_rel_vel = torch.sum(rel_pos * rel_vel, dim=-1)
        rel_vel_norm_sq = torch.sum(rel_vel * rel_vel, dim=-1)

        ttc = torch.full_like(rel_pos_dot_rel_vel, float('inf'))
        valid_mask = (rel_vel_norm_sq > 1e-6) & (rel_pos_dot_rel_vel < 0)
        ttc[valid_mask] = -rel_pos_dot_rel_vel[valid_mask] / rel_vel_norm_sq[valid_mask]
        ttc.fill_diagonal_(float('inf'))
        return ttc

    def forward(self,
                agent_features: torch.Tensor,
                map_features: torch.Tensor,
                agent_types: torch.Tensor,
                agent_positions: torch.Tensor,
                agent_velocities: torch.Tensor,
                agent_headings: torch.Tensor,
                map_positions: torch.Tensor,
                map_orientations: torch.Tensor,
                map_follow_counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_agents = agent_features.size(0)
        num_maps = map_features.size(0)

        if num_agents > 0:
            ttc_matrix = self.compute_ttc(agent_positions.float(), agent_velocities.float())
            finite_ttc = torch.where(torch.isfinite(ttc_matrix), ttc_matrix, torch.full_like(ttc_matrix, 1e6))
            min_ttc, _ = finite_ttc.min(dim=-1)
            min_ttc = torch.clamp(min_ttc, max=10.0)
            agent_risk_scores = torch.exp(-min_ttc / self.agent_risk_temperature)
        else:
            agent_risk_scores = torch.empty(0, device=agent_features.device)

        if num_maps > 0:
            map_follow_scores = torch.clamp(map_follow_counts.float() / self.map_flow_normalizer, 0.0, 1.0)
        else:
            map_follow_scores = torch.empty(0, device=agent_features.device)

        node_embeddings = self.node_feature(
            agent_features=agent_features,
            map_features=map_features,
            agent_types=agent_types,
            agent_risk_scores=agent_risk_scores,
            map_follow_scores=map_follow_scores,
        )

        node_positions = torch.cat([agent_positions, map_positions], dim=0)
        node_headings = torch.cat([agent_headings, map_orientations], dim=0)
        attn_bias = self.attn_bias(node_positions, node_headings)

        x = node_embeddings.unsqueeze(0)
        attn_bias = attn_bias.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)
        x = self.output_proj(x)
        x = x.squeeze(0)

        agent_out = x[1:1 + num_agents]
        map_out = x[1 + num_agents:]
        return agent_out, map_out
