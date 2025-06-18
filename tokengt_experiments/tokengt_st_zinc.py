import os.path as osp
import os
import pickle
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddOrthonormalNodeIdentifiers

import numpy as np
from torch_geometric.transforms.compose import Compose
import wandb
import random
import networkx as nx

from models.add_substructure_instances import AddSubstructureInstances
from tokengt_experiments.exp_models import TokenGTGraphRegression, TokenGTSTSumGraphRegression, TokenGTSTHypGraphRegression


def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()

        if isinstance(model, TokenGTGraphRegression):
            out = model(batch)
        else:
            out = model(batch, batch.substructure_instances)

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
            if isinstance(model, TokenGTGraphRegression):
                out = model(batch)
            else:
                out = model(batch, batch.substructure_instances)

            loss = criterion(out, batch.y.unsqueeze(1)).item()
            total_loss += loss
    return total_loss / len(loader.dataset)


def load_substructures(filepath: str):
    with open(filepath, 'rb') as f:
        subs = pickle.load(f)
        out = []
        for s in subs:
            G = nx.Graph()
            G.add_edges_from(s)
            out.append(G)
        return out


def create_model(config, train_dataset, device, n_substructures):
    """Create model based on architecture type."""
    if config.architecture == "TokenGT":
        return TokenGTGraphRegression(
            dim_node=train_dataset.num_node_features,
            d_p=config.D_P,
            d=config.d,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            include_graph_token=config.include_graph_token,
            is_laplacian_node_ids=config.use_laplacian,
            dim_edge=train_dataset.num_edge_features,
            dropout=config.dropout,
            device=device,
        )
    elif config.architecture == "TokenGTST_Sum":
        return TokenGTSTSumGraphRegression(
            dim_node=train_dataset.num_node_features,
            d_p=config.D_P,
            d=config.d,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            include_graph_token=config.include_graph_token,
            is_laplacian_node_ids=config.use_laplacian,
            dim_edge=train_dataset.num_edge_features,
            dropout=config.dropout,
            device=device,
            n_substructures=n_substructures
        )
    elif config.architecture == "TokenGTST_Hyp":
        return TokenGTSTHypGraphRegression(
            dim_node=train_dataset.num_node_features,
            d_p=config.D_P,
            d=config.d,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            include_graph_token=config.include_graph_token,
            is_laplacian_node_ids=config.use_laplacian,
            dim_edge=train_dataset.num_edge_features,
            dropout=config.dropout,
            device=device,
            n_substructures=n_substructures
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
        "architecture": "TokenGT",
        "dataset": "ZINC_12K",
        "use_features": False,
        "D_P": 64,
        "num_heads": 16,
        "d": 64,
        "num_encoder_layers": 4,
        "dim_feedforward": 64,
        "include_graph_token": True,
        "use_laplacian": False,
        "dropout": 0.1,
        "epochs": 500,
        "lr": 0.001,
        "train_batch_size": 32,
        "substructures_file": "subs_size6",
    }

    run = wandb.init(
        entity="krecharles-university-of-oxford",
        project="TokenGTST_debug",
        config=config,
        mode="disabled"
    )

    config = wandb.config

    substructures = load_substructures(
        f"tokengt_experiments/{config.substructures_file}.pkl")

    transform = Compose([AddOrthonormalNodeIdentifiers(config.D_P, config.use_laplacian),
                         AddSubstructureInstances(substructures)])

    train_dataset = ZINC(f"data/ZINC-lap-{config.substructures_file}-{config.D_P}", subset=True,
                         split="train", pre_transform=transform)
    val_dataset = ZINC(f"data/ZINC-lap-{config.substructures_file}-{config.D_P}", subset=True,
                       split="val", pre_transform=transform)

    if not config.use_features:
        train_dataset.data.x = torch.zeros(
            train_dataset.data.num_nodes, 1)
        val_dataset.data.x = torch.zeros(
            val_dataset.data.num_nodes, 1)
        train_dataset.data.edge_attr = torch.zeros(
            train_dataset.data.num_edges)
        val_dataset.data.edge_attr = torch.zeros(val_dataset.data.num_edges)
        assert train_dataset.data.x[0] == 0
        assert val_dataset.data.x[0] == 0
        assert train_dataset.data.edge_attr[0] == 0
        assert val_dataset.data.edge_attr[0] == 0

    if torch.cuda.is_available():
        train_dataset.cuda()
        val_dataset.cuda()

    print(f"Training with {len(train_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(config, train_dataset, device, len(substructures))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of params: {num_params}")
    print(f"n_substructures: {len(substructures)}")
    run.log({"num_param": num_params})

    criterion = nn.L1Loss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train_loss = get_loss(model, train_loader, criterion)
    val_loss = get_loss(model, val_loader, criterion)
    print(f"Epoch 0: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
    run.log({"train_loss": train_loss, "val_loss": val_loss}, step=1)

    for i in range(2, config.epochs + 2):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = get_loss(model, val_loader, criterion)
        print(f"Epoch {i}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        run.log({"train_loss": train_loss, "val_loss": val_loss}, step=i)

    save_path = f"trained_models/{config.architecture}_{config.dataset}_{config.substructures_file}.pt"
    torch.save(model, save_path)

    run.finish()


if __name__ == "__main__":
    main()
