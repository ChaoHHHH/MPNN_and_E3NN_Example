from typing import Dict, List, Optional
import numpy as np
import torch
from torch_geometric.data import Data
from ase.neighborlist import neighbor_list
from ase import Atoms

from utils import Configuration, Configurations, load_from_xyz


def make_species_map(configs: Configurations) -> Dict[int, int]:
    """统计数据集所有原子序数，构建 Z → species_id 映射"""
    unique_z = sorted(set(z for c in configs for z in c.atomic_numbers))
    return {z: i for i, z in enumerate(unique_z)}


def config_to_graph(
    config: Configuration,
    species_map: Dict[int, int],
    cutoff: float = 5.0,
) -> Data:
    """将单个 Configuration 转换为 PyG Data 对象

    species_map: Z → species_id 映射，用于 one-hot 编码节点特征
    """
    atoms = Atoms(
        numbers=config.atomic_numbers,
        positions=config.positions,
        cell=config.cell,
        pbc=config.pbc if config.pbc is not None else False,
    )

    i, j, d, D = neighbor_list("ijdD", atoms, cutoff=cutoff, self_interaction=False)

    edge_index = torch.tensor(np.stack([i, j], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(np.column_stack([d, D]), dtype=torch.float)

    species_id = np.vectorize(species_map.get)(config.atomic_numbers)
    x = torch.nn.functional.one_hot(
        torch.tensor(species_id, dtype=torch.long),
        num_classes=len(species_map),
    ).float()

    pos = torch.tensor(config.positions, dtype=torch.float)
    y = torch.tensor(config.properties["energy"], dtype=torch.float).view(1)
    force = torch.tensor(config.properties["forces"], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                pos=pos, y=y, force=force)


def configs_to_dataset(
    configs: Configurations,
    species_map: Optional[Dict[int, int]] = None,
    cutoff: float = 5.0,
) -> List[Data]:
    """将 Configurations 列表转换为 PyG Data 列表"""
    if species_map is None:
        species_map = make_species_map(configs)
    return [config_to_graph(c, species_map, cutoff=cutoff) for c in configs]


if __name__ == "__main__":
    configs = load_from_xyz("stru.xyz")
    species_map = make_species_map(configs)
    print(species_map)
    d = config_to_graph(configs[0], species_map)
    print(d.x.shape)
    print(d.edge_index.shape)
