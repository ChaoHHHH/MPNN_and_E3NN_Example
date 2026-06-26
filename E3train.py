import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from utils import load_from_xyz
from graph import config_to_graph, make_species_map, configs_to_dataset
from E3model import E3Model


def train():
    device = torch.device("cuda")
    print(f"Device: {device}")

    print("Loading data...")
    configs = load_from_xyz("stru.xyz")
    test_config = configs[0]
    train_configs = configs[1:]
    print(f"Train: {len(train_configs)}, Test: 1")

    species_map = make_species_map(configs)
    dataset = configs_to_dataset(train_configs, species_map=species_map, cutoff=5.0)
    test_data = config_to_graph(test_config, species_map=species_map, cutoff=5.0)

    model = E3Model(n_species=len(species_map)).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    loader = DataLoader(dataset, batch_size=24, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    force_weight = 1000
    n_epochs = 300

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            pred_energy, pred_force = model(batch)

            loss_energy = loss_fn(pred_energy, batch.y.squeeze(-1))
            loss_force = loss_fn(pred_force, batch.force)
            loss = loss_energy + force_weight * loss_force

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  Loss: {loss.item():.4f}  "
                  f"E: {loss_energy.item():.4f}  F: {loss_force.item():.6f}")

    model.eval()
    with torch.no_grad():
        test_batch = test_data.to(device)
        pred_energy, pred_force = model(test_batch)
        true_energy = test_batch.y.squeeze(-1)
        true_force = test_batch.force

        force_mae = (pred_force - true_force).abs().mean()

        print(f"\nTest result:")
        print(f"  Pred energy: {pred_energy.item():.4f}")
        print(f"  True energy: {true_energy.item():.4f}")
        print(f"  Energy error: {(pred_energy - true_energy).item():.4f}")
        print(f"  Force MAE: {force_mae.item():.6f}")

    return model


if __name__ == "__main__":
    train()
