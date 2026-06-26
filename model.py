from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class MessageLayer(MessagePassing):
    def __init__(self, d_hidden: int, d_edge: int):
        super().__init__(aggr="sum")
        self.msg_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):
        return self.msg_mlp(torch.cat([x_j, x_i, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class SimpleMPNN(nn.Module):
    def __init__(
        self,
        n_species: int,
        d_hidden: int = 64,
        d_edge: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.embed = nn.Linear(n_species, d_hidden)
        self.edge_net = nn.Sequential(
            nn.Linear(4, d_edge),
            nn.ReLU(),
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([
            MessageLayer(d_hidden, d_edge) for _ in range(n_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, data):
        x = self.embed(data.x)
        edge_attr = self.edge_net(data.edge_attr)
        for layer in self.layers:
            x = x + layer(x, data.edge_index, edge_attr)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        out = scatter(x, batch, dim=0, reduce="sum")
        return self.readout(out).squeeze(-1)
