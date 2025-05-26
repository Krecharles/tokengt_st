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
from models.token_gt_st import TokenGTST


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
        n_substructures
    ):
        super().__init__()
        self._token_gt = TokenGTST(
            dim_node=dim_node,
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dim_edge=dim_edge,
            is_laplacian_node_ids=is_laplacian_node_ids,
            include_graph_token=include_graph_token,
            dropout=dropout,
            device=device,
            n_substructures=n_substructures
        )
        self.lm = nn.Linear(d, 1, device=device)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
        node_ids: Tensor,
        substructure_instances: list[list[list[int]]],
    ):
        _, graph_emb = self._token_gt(x, edge_index, edge_attr, ptr, batch,
                                      node_ids, substructure_instances)
        return self.lm(graph_emb)


def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        out = model(
            batch.x.float(),
            batch.edge_index,
            batch.edge_attr.unsqueeze(1).float(),
            batch.ptr,
            batch.batch,
            batch.node_ids,
            batch.substructure_instances
        )
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
            out = model(
                batch.x.float(),
                batch.edge_index,
                batch.edge_attr.unsqueeze(1).float(),
                batch.ptr,
                batch.batch,
                batch.node_ids,
                batch.substructure_instances
            )
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
    # d = head_dim * num_heads
    # num_encoder_layers = 12
    # dim_feedforward = d # TokenGT: 768 = 32 * 24
    # dropout=0.1

    config = {
        "architecture": "TokenGTST",
        "dataset": "ZINC_12K",
        "D_P": 8,
        # "head_dim": 4,
        "num_heads": 4,
        "d": 64,
        "num_encoder_layers": 2,
        "dim_feedforward": 128,
        "include_graph_token": True,
        "use_laplacian": True,
        "dropout": 0.1,
        "epochs": 20,
        "lr": 0.001
    }

    run = wandb.init(
        entity="krecharles-university-of-oxford",
        project="TokenGT_ST",
        config=config,
        # mode="disabled"
    )

    config = wandb.config

    substructures = load_substructures("tokengt_zinc/substructures.pkl")
    transform = Compose([AddOrthonormalNodeIdentifiers(config.D_P, config.use_laplacian),
                         AddSubstructureInstances(substructures)])

    train_dataset = ZINC(f"data/ZINC-lap-subs-{config.D_P}", subset=True,
                         split="train", pre_transform=transform)
    val_dataset = ZINC(f"data/ZINC-lap-subs-{config.D_P}", subset=True,
                       split="val", pre_transform=transform)

    if torch.cuda.is_available():
        train_dataset.cuda()
        val_dataset.cuda()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TokenGTGraphRegression(
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
        n_substructures=len(substructures)
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

    for i in range(2, config.epochs + 2):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = get_loss(model, val_loader, criterion)
        print(f"Epoch {i}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        run.log({"train_loss": train_loss, "val_loss": val_loss}, step=i)

    run.finish()


if __name__ == "__main__":
    main()
