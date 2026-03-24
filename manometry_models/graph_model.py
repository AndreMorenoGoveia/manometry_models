from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class SparseGraphAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_nodes: int,
        graph_radius: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.graph_radius = graph_radius
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        self.representation_scale = nn.Parameter(torch.ones(num_heads))
        self.relative_position_bias = nn.Embedding(2 * num_nodes - 1, num_heads)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = nodes.shape
        normalized = self.norm1(nodes)

        qkv = self.qkv(normalized)
        qkv = qkv.view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        attention_logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        representation = F.normalize(normalized, dim=-1)
        representation_corr = torch.matmul(representation, representation.transpose(-2, -1))
        attention_logits = attention_logits + (
            representation_corr.unsqueeze(1) * self.representation_scale.view(1, self.num_heads, 1, 1)
        )

        positions = torch.arange(num_nodes, device=nodes.device)
        relative = positions[:, None] - positions[None, :]
        relative_bias = self.relative_position_bias(relative + num_nodes - 1)
        attention_logits = attention_logits + relative_bias.permute(2, 0, 1).unsqueeze(0)

        if self.graph_radius >= 0:
            sparse_mask = relative.abs() <= self.graph_radius
            sparse_mask = sparse_mask.to(device=nodes.device)
            attention_logits = attention_logits.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attention = torch.softmax(attention_logits, dim=-1)
        attention = self.attn_dropout(attention)

        updated = torch.matmul(attention, value)
        updated = updated.transpose(1, 2).reshape(batch_size, num_nodes, self.hidden_dim)
        updated = self.proj_dropout(self.proj(updated))

        nodes = nodes + updated
        nodes = nodes + self.mlp(self.norm2(nodes))
        return nodes


class WangCVPGAT(nn.Module):
    """Wang-inspired CVP-GAT for HRM image classification.

    The paper organizes the swallow into six vigor regions. Here we mirror that
    by pooling the CNN feature map into six vertical graph nodes and using a
    sparse graph attention stack that combines representation correlation and
    relative-position correlation.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool,
        dropout: float,
        num_graph_nodes: int,
        graph_temporal_bins: int,
        graph_hidden_dim: int,
        graph_num_heads: int,
        graph_num_layers: int,
        graph_radius: int,
    ) -> None:
        super().__init__()
        if num_graph_nodes < 2:
            raise ValueError("num_graph_nodes must be at least 2.")
        if graph_temporal_bins < 1:
            raise ValueError("graph_temporal_bins must be at least 1.")
        if graph_hidden_dim < 8:
            raise ValueError("graph_hidden_dim must be at least 8.")
        if graph_num_layers < 1:
            raise ValueError("graph_num_layers must be at least 1.")
        if graph_num_heads < 1:
            raise ValueError("graph_num_heads must be at least 1.")

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.num_graph_nodes = num_graph_nodes
        self.graph_temporal_bins = graph_temporal_bins
        self.feature_pool = nn.AdaptiveAvgPool2d((num_graph_nodes, graph_temporal_bins))
        self.node_projection = nn.Sequential(
            nn.LayerNorm(512 * graph_temporal_bins),
            nn.Linear(512 * graph_temporal_bins, graph_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.node_position = nn.Parameter(torch.zeros(1, num_graph_nodes, graph_hidden_dim))
        self.graph_blocks = nn.ModuleList(
            [
                SparseGraphAttentionBlock(
                    hidden_dim=graph_hidden_dim,
                    num_heads=graph_num_heads,
                    num_nodes=num_graph_nodes,
                    graph_radius=graph_radius,
                    dropout=dropout,
                )
                for _ in range(graph_num_layers)
            ]
        )
        self.readout_query = nn.Parameter(torch.zeros(graph_hidden_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(graph_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden_dim, num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.node_position, std=0.02)
        nn.init.normal_(self.readout_query, std=1.0 / math.sqrt(self.readout_query.numel()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = self.feature_pool(features)
        nodes = pooled.permute(0, 2, 1, 3).reshape(x.size(0), self.num_graph_nodes, -1)
        nodes = self.node_projection(nodes)
        nodes = nodes + self.node_position

        for block in self.graph_blocks:
            nodes = block(nodes)

        readout_scores = torch.matmul(nodes, self.readout_query)
        readout_attention = torch.softmax(readout_scores, dim=1).unsqueeze(-1)
        pooled_nodes = torch.sum(nodes * readout_attention, dim=1)
        return self.classifier(pooled_nodes)
