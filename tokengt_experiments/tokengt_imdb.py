import os.path as osp
import os
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TokenGT
from torch_geometric.transforms import AddOrthonormalNodeIdentifiers, Compose

import numpy as np
import wandb
import random


class TokenGTGraphClassification(nn.Module):
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
        dropout,
        device,
        num_classes,
    ):
        super().__init__()
        self._token_gt = TokenGT(
            dim_node=dim_node,
            dim_edge=None,
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
        self.lm = nn.Linear(d, num_classes, device=device)

    def forward(self, batch):
        _, graph_emb = self._token_gt(batch.x.float(),
                                      batch.edge_index,
                                      None,
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids)
        return self.lm(graph_emb)


class AddZeroFeatures:
    def __call__(self, data):
        if data.x is None:
            data.x = torch.zeros((data.num_nodes, 1))
        return data


def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(loader.dataset)


def get_loss_and_accuracy(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            loss = criterion(out, batch.y).item()
            total_loss += loss
            total_acc += (out.argmax(dim=1) == batch.y).sum().item()
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

    config = {
        "architecture": "TokenGT",
        "dataset": "IMDB-BINARY",
        "use_features": False,
        "D_P": 64,
        "num_heads": 8,
        "d": 32,
        "num_encoder_layers": 3,
        "dim_feedforward": 32,
        "include_graph_token": True,
        "use_laplacian": False,
        "dropout": 0.1,
        "epochs": 30,
        "lr": 0.001,
        "train_batch_size": 32,
    }

    run = wandb.init(
        entity="krecharles-university-of-oxford",
        project="TokenGT_imdb",
        config=config,
        mode="disabled"
    )

    config = wandb.config

    transform = Compose([AddZeroFeatures(), AddOrthonormalNodeIdentifiers(
        config.D_P, config.use_laplacian)])

    path = osp.join(osp.realpath(os.getcwd()),
                    "data", f"IMDB-{config.use_laplacian}-{config.D_P}")
    dataset = TUDataset(root=path,
                        name='IMDB-BINARY', pre_transform=transform)

    # Split dataset into train/val/test
    num_graphs = len(dataset)
    indices = list(range(num_graphs))
    random.shuffle(indices)

    train_size = int(0.8 * num_graphs)
    val_size = int(0.1 * num_graphs)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    if torch.cuda.is_available():
        train_dataset.cuda()
        val_dataset.cuda()
        test_dataset.cuda()

    train_loader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TokenGTGraphClassification(
        dim_node=train_dataset.num_node_features,
        d_p=config.D_P,
        d=config.d,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        include_graph_token=config.include_graph_token,
        is_laplacian_node_ids=config.use_laplacian,
        dropout=config.dropout,
        num_classes=train_dataset.num_classes,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of params: {num_params}")
    run.log({"num_param": num_params})

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Initial evaluation
    train_loss, train_acc = get_loss_and_accuracy(
        model, train_loader, criterion)
    val_loss, val_acc = get_loss_and_accuracy(model, val_loader, criterion)
    print(
        f"Epoch 0: train_loss={train_loss:.5f} train_acc={train_acc:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.5f}")
    run.log({"train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc}, step=1)

    best_val_acc = val_acc

    for i in range(2, config.epochs + 2):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer)
        val_loss, val_acc = get_loss_and_accuracy(model, val_loader, criterion)

        print(
            f"Epoch {i}: train_loss={train_loss:.5f} train_acc={train_acc:.5f} val_loss={val_loss:.5f} val_acc={val_acc:.5f}")
        run.log({"train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc}, step=i)

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            run.log({"best_val_acc": best_val_acc}, step=i)

    # Final test evaluation
    test_loss, test_acc = get_loss_and_accuracy(model, test_loader, criterion)
    print(f"Final test accuracy: {test_acc:.5f}")
    run.log({"test_acc": test_acc})

    run.finish()


if __name__ == "__main__":
    main()
