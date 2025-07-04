import os.path as osp
import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddOrthonormalNodeIdentifiers

import numpy as np
import wandb
import random
from sklearn.model_selection import train_test_split

from tokengt_experiments.exp_models import TokenGTGraphRegression, GCNGraphRegression


def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    device = next(model.parameters()).device
    for batch in loader:
        batch = batch.to(device)  # Move batch to GPU
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
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)  # Move batch to GPU
            out = model(batch)
            loss = criterion(out, batch.y.unsqueeze(1)).item()
            total_loss += loss
    return total_loss / len(loader.dataset)


def add_zero_features(data):
    if not hasattr(data, 'x') or data.x is None:
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        data.edge_attr = torch.zeros(
            (data.edge_index.size(1)), dtype=torch.float)
    return data


def create_model(config, train_dataset, device):
    """Create model based on architecture type."""
    if config.architecture == "TokenGT":
        return TokenGTGraphRegression(
            dim_node=1,
            dim_edge=1,
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
    elif config.architecture == "GCN":
        return GCNGraphRegression(
            dim_node=1,
            hidden_channels=config.d,
            num_layers=config.num_encoder_layers,
            dropout=config.dropout,
            device=device,
        )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    config = {
        "architecture": "TokenGT",  # Options: "TokenGT", "GCN"
        "dataset": "synthetic_cycle_dataset.pt",
        "D_P": 32,
        "num_heads": 16,
        "d": 32,
        "num_encoder_layers": 3,
        "dim_feedforward": 32,
        "include_graph_token": True,
        "use_laplacian": False,
        "dropout": 0.1,
        "epochs": 500,
        "lr": 0.001,
        "batch_size": 32,
        "weight_decay": 0.01,  # Added weight decay for regularization
    }

    run = wandb.init(
        entity="krecharles-university-of-oxford",
        project="TokenGT_synthetic_cycle_ds",
        config=config,
        # mode="disabled"
    )

    config = wandb.config

    transform = AddOrthonormalNodeIdentifiers(
        config.D_P, config.use_laplacian)
    # note: use pre_transform (avoid unnecessary duplicate eigenvector calculation)
    train_dataset = torch.load("structural_dataset/synthetic_cycle_dataset.pt", weights_only=False)
    train_dataset = [transform(add_zero_features(data))
                     for data in train_dataset]

    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=0.2, random_state=42, shuffle=True
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(config, train_dataset, device)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of params: {num_params}")
    print(f"Architecture: {config.architecture}")
    run.log({"num_param": num_params})

    criterion = nn.L1Loss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_loss = get_loss(model, train_loader, criterion)
    val_loss = get_loss(model, val_loader, criterion)
    print(f"Epoch 0: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
    run.log({"train_loss": train_loss, "val_loss": val_loss, "architecture": config.architecture}, step=1)

    for i in range(2, config.epochs + 2):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = get_loss(model, val_loader, criterion)
        print(f"Epoch {i}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        run.log({"train_loss": train_loss, "val_loss": val_loss, "architecture": config.architecture}, step=i)

    run.finish()


if __name__ == "__main__":
    main()
