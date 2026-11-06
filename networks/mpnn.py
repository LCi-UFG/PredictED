"""Tier-1 MPNN architecture required by the released prediction checkpoint."""

import torch
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge, global_add_pool

from activation import get_activation
from layers import MPNNLayer


class MPNNet(nn.Module):
    def __init__(self, node_dim, edge_dim, agg_hidden_dims, num_agg_layers,
                 lin_hidden_dims, num_lin_layers, activation, dropout_rate,
                 num_tasks):
        super().__init__()
        dims = [agg_hidden_dims[i] for i in range(num_agg_layers)]
        input_dims = [node_dim] + dims[:-1]
        self.agg_layers = nn.ModuleList([
            MPNNLayer(input_dims[i], dims[i], edge_dim, dropout_rate)
            for i in range(num_agg_layers)
        ])
        self.jk = JumpingKnowledge(mode="cat")
        self.virtualnode_embedding = nn.ParameterList([
            nn.Parameter(torch.zeros(1, dim)) for dim in input_dims
        ])
        self.virtualnode_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim),
            )
            for input_dim, output_dim in zip(input_dims, dims)
        ])
        self.lin_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sum(dims) if i == 0 else lin_hidden_dims[i - 1],
                          lin_hidden_dims[i]),
                get_activation(activation),
                nn.Dropout(dropout_rate),
            )
            for i in range(num_lin_layers)
        ])
        embedding_dim = lin_hidden_dims[-1]
        self.embedding_layer = nn.Linear(embedding_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, num_tasks)

    def forward(self, data, return_penultimate=False):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        num_graphs = batch.max().item() + 1
        virtual_nodes = [parameter.expand(num_graphs, -1)
                         for parameter in self.virtualnode_embedding]
        layer_outputs = []
        for i, layer in enumerate(self.agg_layers):
            x = layer(x + virtual_nodes[i][batch], edge_index, edge_attr, batch)
            layer_outputs.append(x)
            pooled = global_add_pool(x, batch)
            virtual_nodes[i] = virtual_nodes[i] + self.virtualnode_mlp[i](pooled)
        x = global_add_pool(self.jk(layer_outputs), batch)
        for layer in self.lin_layers:
            x = layer(x)
        embedding = self.embedding_layer(x)
        if return_penultimate:
            return embedding
        return self.output_layer(embedding)
