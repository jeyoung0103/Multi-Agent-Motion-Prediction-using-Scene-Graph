"""
QCNet Agent Encoder with Graphormer integration
Replaces Agent-Agent Attention with Graphormer for better global relationship modeling
"""

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import weight_init
from utils import wrap_angle
from modules.graphormer_components import GraphormerGraphEncoder


class QCNetAgentEncoderWithGraphormer(nn.Module):
    """
    QCNet Agent Encoder with Graphormer integration
    Replaces Agent-Agent Attention with Graphormer for global relationship modeling
    """

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 # Graphormer parameters
                 use_graphormer: bool = True,
                 num_agent_types: int = 5,
                 num_speed_bins: int = 10,
                 num_spatial_bins: int = 20,
                 num_ttc_bins: int = 10,
                 num_edge_types: int = 5,
                 **kwargs) -> None:
        super(QCNetAgentEncoderWithGraphormer, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_agent_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        
        # Graphormer integration flag
        self.use_graphormer = use_graphormer

        if dataset == 'ETRI_Dataset':
            input_dim_x_a = 4  # agent dim
            input_dim_r_t = 4  # time dim
            input_dim_r_pl2a = 3  # polygon to agent dim
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'ETRI_Dataset':
            self.type_a_emb = nn.Embedding(10, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
            
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        
        # Temporal attention layers (unchanged)
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        
        # Map-Agent attention layers (unchanged)
        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(self.num_layers)]
        )
        
        # Graphormer for Agent-Agent relationships (replaces a2a_attn_layers)
        if self.use_graphormer:
            self.graphormer_encoder = GraphormerGraphEncoder(
                num_heads=num_heads,
                num_agent_types=num_agent_types,
                num_speed_bins=num_speed_bins,
                num_spatial_bins=num_spatial_bins,
                num_ttc_bins=num_ttc_bins,
                num_edge_types=num_edge_types,
                hidden_dim=hidden_dim,
                num_layers=num_agent_layers,
                dropout=dropout
            )
        else:
            # Fallback to original Agent-Agent attention
            self.a2a_attn_layers = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                bipartite=False, has_pos_emb=True) for _ in range(self.num_layers)]
            )
            
        self.apply(weight_init)

    def forward(self, map_enc_x_pl: torch.Tensor,
               mask: torch.Tensor,
               pos_a: torch.Tensor,
               motion_vector_a: torch.Tensor,
               head_a: torch.Tensor,
               pos_pl: torch.Tensor,
               orient_pl: torch.Tensor,
               vel: torch.Tensor,
               agent_type: torch.Tensor,
               batch_s: torch.Tensor,
               batch_pl: torch.Tensor):

        categorical_embs = [self.type_a_emb(agent_type)
                            .repeat_interleave(repeats=self.num_historical_steps,
                                               dim=0), ]  # (num_agent x obs_len) x 128

        motion_vector_a = motion_vector_a
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        if self.dataset == 'ETRI_Dataset':
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                 torch.norm(vel[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        x_a = x_a.view(-1, x_a.size(-1)).float()
        x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)

        # Temporal attention processing
        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = r_t.float()
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        # Map-Agent attention processing
        base_pos_pl = pos_pl  # keep original (no time repeat) for graphormer
        base_orient_pl = orient_pl

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1)
        orient_pl = orient_pl.repeat(self.num_historical_steps)

        pos_s = pos_s.double()
        pos_pl = pos_pl.double()
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius, batch_x=batch_s, batch_y=batch_pl,
                                 max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a.float(), categorical_embs=None)

        # Apply temporal and map-agent attention layers
        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
            x_a = x_a.reshape(-1, self.num_historical_steps,
                              self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a = self.pl2a_attn_layers[i]((map_enc_x_pl.transpose(0, 1).reshape(-1, self.hidden_dim), x_a), r_pl2a,
                                           edge_index_pl2a)
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)

        # Agent-Agent relationship processing
        if self.use_graphormer:
            # Use Graphormer for global agent/map relationships (agent + map tokens)
            x_a_last = x_a[:, -1, :]  # [num_agents, hidden_dim]
            map_last = map_enc_x_pl[:, -1, :]  # [num_maps, hidden_dim]
            pos_a_last = pos_a[:, -1, :]  # [num_agents, 2]
            vel_a_last = vel[:, -1, :]  # [num_agents, 2]
            head_a_last = head_a[:, -1]  # [num_agents]

            if agent_type.dim() == 2:
                agent_type_last = agent_type[:, -1]
            else:
                agent_type_last = agent_type

            # Recover per-graph batches for last timestep
            agent_batch_last = None
            map_batch_last = None
            if batch_s is not None:
                agent_batch_last = batch_s.view(self.num_historical_steps, -1)[-1]
            if batch_pl is not None:
                map_batch_last = batch_pl.view(self.num_historical_steps, -1)[-1]
            if agent_batch_last is None:
                agent_batch_last = torch.zeros(pos_a_last.size(0), dtype=torch.long, device=pos_a_last.device)
            if map_batch_last is None:
                map_batch_last = torch.zeros(base_pos_pl.size(0), dtype=torch.long, device=base_pos_pl.device)

            # Compute per-map follow counts (number of nearby agents)
            map_follow_counts = torch.zeros(base_pos_pl.size(0), device=base_pos_pl.device)
            if base_pos_pl.size(0) > 0 and pos_a_last.size(0) > 0:
                edge_index_pl2a_last = radius(
                    x=pos_a_last[:, :2].double(),
                    y=base_pos_pl[:, :2].double(),
                    r=self.pl2a_radius,
                    batch_x=agent_batch_last,
                    batch_y=map_batch_last,
                    max_num_neighbors=300)
                if edge_index_pl2a_last.numel() > 0:
                    ones = torch.ones(edge_index_pl2a_last.size(1), device=map_follow_counts.device)
                    map_follow_counts.scatter_add_(0, edge_index_pl2a_last[0], ones)

            x_a_enhanced, x_pl_enhanced = self.graphormer_encoder(
                agent_features=x_a_last,
                map_features=map_last,
                agent_types=agent_type_last,
                agent_positions=pos_a_last,
                agent_velocities=vel_a_last,
                agent_headings=head_a_last,
                map_positions=base_pos_pl,
                map_orientations=base_orient_pl,
                map_follow_counts=map_follow_counts
            )

            # Replace the last timestep with enhanced features (agent + map)
            x_a[:, -1, :] = x_a_enhanced
            map_enc_x_pl[:, -1, :] = x_pl_enhanced
            
        else:
            # Fallback to original Agent-Agent attention
            pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
            head_s = head_a.transpose(0, 1).reshape(-1)
            head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
            mask_s = mask.transpose(0, 1).reshape(-1)
            
            from torch_geometric.utils import subgraph
            edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                          max_num_neighbors=300)
            edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
            rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
            rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
            r_a2a = torch.stack(
                [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
                 rel_head_a2a], dim=-1)
            r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a.float(), categorical_embs=None)
            
            # Apply Agent-Agent attention
            for i in range(self.num_layers):
                x_a = x_a.reshape(-1, self.hidden_dim)
                x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
                x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)

        outputs = {'x_a': x_a}
        if self.use_graphormer:
            outputs['x_pl'] = map_enc_x_pl
        return outputs
