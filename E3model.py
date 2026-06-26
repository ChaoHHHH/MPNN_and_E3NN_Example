import torch
import torch.nn as nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from torch_scatter import scatter


class RadialBasis(nn.Module):
    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return soft_one_hot_linspace(
            d, 0.0, self.cutoff, self.num_basis,
            basis="cosine", cutoff=True,
        )


class EquivariantConv(nn.Module):
    def __init__(self, node_irreps_in, node_irreps_out, sh_irreps, rbf_dim: int):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(
            node_irreps_in, sh_irreps, node_irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.rbf_mlp = nn.Linear(rbf_dim, self.tp.weight_numel)

    def forward(self, x, edge_index, sh, rbf):
        src, dst = edge_index
        x_j = x[src]
        rbf_weight = self.rbf_mlp(rbf)
        message = self.tp(x_j, sh, weight=rbf_weight)
        out = scatter(message, dst, dim=0, dim_size=x.size(0), reduce="sum")
        return out


class ConvBlock(nn.Module):
    def __init__(self, node_irreps_in, node_irreps_out, sh_irreps, rbf_dim: int):
        super().__init__()
        self.conv = EquivariantConv(node_irreps_in, node_irreps_out, sh_irreps, rbf_dim)

        irreps_out = o3.Irreps(node_irreps_out)
        scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
        non_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])

        if len(non_scalars) > 0:
            gates = o3.Irreps([(mul, "0e") for mul, _ in non_scalars])
            pre_gate_irreps = scalars + gates + non_scalars
            self.linear = o3.Linear(node_irreps_out, pre_gate_irreps)
            self.gate = Gate(
                scalars, [nn.SiLU()] * len(scalars),
                gates, [nn.SiLU()] * len(gates),
                non_scalars,
            )
            self.out_irreps = scalars + non_scalars
        else:
            self.linear = o3.Linear(node_irreps_out, node_irreps_out)
            self.gate = nn.SiLU()
            self.out_irreps = irreps_out

        self.use_skip = (o3.Irreps(node_irreps_in) == self.out_irreps)

    def forward(self, x, edge_index, sh, rbf):
        x_new = self.conv(x, edge_index, sh, rbf)
        x_new = self.linear(x_new)
        x_new = self.gate(x_new)
        if self.use_skip:
            x_new = x_new + x
        return x_new


class E3Model(nn.Module):
    def __init__(self, n_species: int, num_rbf: int = 8, cutoff: float = 5.0, l_max: int = 2):
        super().__init__()
        self.cutoff = cutoff

        self.embed = o3.Linear(
            o3.Irreps(f"{n_species}x0e"),
            o3.Irreps("16x0e"),
        )

        self.sh_irreps = o3.Irreps("1x0e + 1x1o + 1x2e")
        self.radial_basis = RadialBasis(num_rbf, cutoff)

        self.conv1 = ConvBlock("16x0e", "16x0e + 8x1o", self.sh_irreps, num_rbf)
        self.conv2 = ConvBlock("16x0e + 8x1o", "16x0e + 8x1o + 8x2e", self.sh_irreps, num_rbf)
        self.conv3 = ConvBlock("16x0e + 8x1o + 8x2e", "8x0e + 4x1o + 4x2e", self.sh_irreps, num_rbf)

        feat_irreps = o3.Irreps("8x0e + 4x1o + 4x2e")
        self.head = o3.Linear(feat_irreps, "1x0e + 1x1o")

    def forward(self, data):
        edge_index = data.edge_index
        src, dst = edge_index

        vec = data.pos[src] - data.pos[dst]
        dist = torch.norm(vec, dim=-1)

        rbf = self.radial_basis(dist)
        sh = o3.spherical_harmonics(range(3), vec, normalize=True)

        x = self.embed(data.x)
        x = self.conv1(x, edge_index, sh, rbf)
        x = self.conv2(x, edge_index, sh, rbf)
        x = self.conv3(x, edge_index, sh, rbf)

        out = self.head(x)
        energy_s = out[:, :1]
        forces = out[:, 1:]

        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        energy = scatter(energy_s, batch, dim=0, reduce="sum").squeeze(-1)

        return energy, forces
