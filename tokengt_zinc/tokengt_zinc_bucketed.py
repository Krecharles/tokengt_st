import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import wandb
import random
import networkx as nx

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TokenGT
from torch_geometric.transforms import AddOrthonormalNodeIdentifiers
from torch_geometric.utils.convert import to_networkx


class TokenGTGraphRegression(nn.Module):
    def __init__(
        self,
        dim_node,
        d_p,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        include_graph_token,
        is_laplacian_node_ids,
        dim_edge,
        dropout,
        device,
    ):
        super().__init__()
        self._token_gt = TokenGT(
            dim_node=dim_node,
            dim_edge=dim_edge,
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            is_laplacian_node_ids=is_laplacian_node_ids,
            include_graph_token=include_graph_token,
            dropout=dropout,
            device=device,
        )
        self.lm = nn.Linear(d, 1, device=device)

    def forward(self, batch):
        _, graph_emb = self._token_gt(batch.x.float(),
                                      batch.edge_index,
                                      batch.edge_attr.unsqueeze(1).float(),
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids)
        return self.lm(graph_emb)


def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(loader.dataset)


def get_loss(model, loader, criterion) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            loss = criterion(out, batch.y.unsqueeze(1)).item()
            total_loss += loss
    return total_loss / len(loader.dataset)


def create_cycle_buckets(dataset):
    """Create buckets of graphs based on their number of cycles."""
    cycle_buckets = {i: [] for i in range(11)}  # 0-9 cycles, and 10+ cycles

    for i, graph_data in enumerate(dataset):
        G = to_networkx(graph_data, to_undirected=True)

        cycles = [cycle for cycle in nx.simple_cycles(G) if len(cycle) >= 6]
        num_cycles = len(cycles)

        bucket = min(num_cycles, 10)
        cycle_buckets[bucket].append(graph_data)

    # Print distribution summary
    print("\nCycle Distribution Summary:")
    print(f"Total graphs analyzed: {len(dataset)}")
    for i in range(11):
        if len(cycle_buckets[i]) > 0:
            print(
                f"Graphs with {i if i < 10 else '10+'} cycles: {len(cycle_buckets[i])}")

    bucket_loaders = {}
    for i in range(11):
        bucket_loaders[i] = DataLoader(cycle_buckets[i], batch_size=128)

    return bucket_loaders


def evaluate_cycle_buckets(model, bucket_loaders, device, epoch, criterion):
    """Evaluate model performance on each cycle bucket."""
    model.eval()

    with torch.no_grad():
        for bucket, loader in bucket_loaders.items():
            total_loss = 0.0
            total_samples = 0

            for batch in loader:
                batch = batch.to(device)
                predictions = model(batch)
                loss = criterion(predictions, batch.y.unsqueeze(1))
                total_loss += loss.item()
                total_samples += len(batch.y)

            mae = total_loss / total_samples
            print(f"Cycle {bucket} MAE: {mae:.4f}")

            wandb.log({
                f"cycle_{bucket}_mae": mae,
            }, step=epoch)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    # TokenGT:
    # initial_lr = 0.001
    # lr_reduce_factor = 0.5
    # minimum_lr = 10^-5
    # patience = 10
    # D_P = 16  # TokenGT: 16 for Lap, 64 for ORF
    # head_dim = 24
    # num_heads = 32
    # num_encoder_layers = 12
    # dim_feedforward = d # TokenGT: 768 = 32 * 24
    # dropout=0.1

    config = {
        "architecture": "TokenGT",
        "dataset": "ZINC_12K",
        "D_P": 64,
        # "head_dim": 4,
        "num_heads": 16,
        "d": 64,
        "num_encoder_layers": 4,
        "dim_feedforward": 64,
        "include_graph_token": True,
        "use_laplacian": False,
        "dropout": 0.1,
        "epochs": 200,
        "lr": 0.001,
        "train_batch_size": 32,
    }

    run = wandb.init(
        entity="krecharles-university-of-oxford",
        project="TokenGT",
        config=config,
        mode="disabled"
    )

    config = wandb.config

    transform = AddOrthonormalNodeIdentifiers(
        config.D_P, config.use_laplacian)
    path = osp.join(osp.realpath(os.getcwd()),
                    "data", f"ZINC-ort-{config.D_P}")
    # note: use pre_transform (avoid unnecessary duplicate eigenvector calculation)
    train_dataset = ZINC(path, subset=True, split="train",
                         pre_transform=transform)
    val_dataset = ZINC(path, subset=True, split="val", pre_transform=transform)

    if torch.cuda.is_available():
        train_dataset.cuda()
        val_dataset.cuda()

    train_loader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    print("\nCreating cycle buckets for validation set...")
    bucket_loaders = create_cycle_buckets(val_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TokenGTGraphRegression(
        dim_node=train_dataset.num_node_features,
        dim_edge=train_dataset.num_edge_features,
        d_p=config.D_P,
        d=config.d,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        include_graph_token=config.include_graph_token,
        is_laplacian_node_ids=config.use_laplacian,
        dropout=config.dropout,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of params: {num_params}")
    run.log({"num_param": num_params})

    criterion = nn.L1Loss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train_loss = get_loss(model, train_loader, criterion)
    val_loss = get_loss(model, val_loader, criterion)
    print(f"Epoch 0: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
    run.log({"train_loss": train_loss, "val_loss": val_loss}, step=1)

    print("\nEvaluating cycle buckets at epoch 0...")
    evaluate_cycle_buckets(model, bucket_loaders, device, 1, criterion)

    for i in range(2, config.epochs + 2):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = get_loss(model, val_loader, criterion)
        print(f"Epoch {i}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        run.log({"train_loss": train_loss, "val_loss": val_loss}, step=i)

        evaluate_cycle_buckets(model, bucket_loaders, device, i, criterion)

    # Save the model
    model_path = osp.join(osp.realpath(os.getcwd()),
                          "trained_models", f"tokengt_zinc_{run.name}.pt")
    os.makedirs(osp.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    run.finish()


if __name__ == "__main__":
    main()
