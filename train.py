import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from utils import load_from_xyz
from graph import configs_to_dataset, config_to_graph, make_species_map
from model import SimpleMPNN


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

    n_species = len(species_map)
    model = SimpleMPNN(n_species=n_species).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    n_epochs = 1000
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch.y.squeeze(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        test_batch = test_data.to(device)
        pred_energy = model(test_batch)
        true_energy = test_batch.y.squeeze(-1)
        print(f"\nTest prediction:")
        print(f"  Predicted energy: {pred_energy.item():.4f}")
        print(f"  Actual energy:    {true_energy.item():.4f}")
        print(f"  Error:            {(pred_energy - true_energy).item():.4f}")

    return model


if __name__ == "__main__":
    train()
