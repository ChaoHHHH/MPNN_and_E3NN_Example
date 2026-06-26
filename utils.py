from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from ase.io import read
from ase import Atoms
import ase

Positions = np.ndarray
Cell = np.ndarray
Pbc = tuple

@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    properties: Dict[str, Any]
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

Configurations = List[Configuration]

def load_from_xyz(file_path: str) -> Configurations:
    '''
    读取 xyz 文件，生成 Configurations = List[Configuration]
    '''
    atoms_list : List[Atoms] = read(filename=file_path, index=":", format="extxyz")
    configs = []
    for atoms in atoms_list:
        atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
        pbc = tuple(atoms.get_pbc().tolist())
        cell = np.array(atoms.get_cell())
        positions = atoms.get_positions()
        properties = {}
        properties["energy"] = atoms.get_total_energy()
        properties["forces"] = atoms.arrays.get("force")
        configs.append(Configuration(
            atomic_numbers=atomic_numbers,
            positions=positions,
            properties=properties,
            cell=cell,
            pbc=pbc,
        ))
    return configs

if __name__ == "__main__":
    xyz_file = "./stru.xyz"
    configs = load_from_xyz(xyz_file)
    # print(f"Loaded {len(configs)} configurations")
    # print(f"Forces shape: {configs[0].properties['forces'].shape}")
    print(len(set(configs[0].atomic_numbers)))