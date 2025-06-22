from itertools import chain
import os.path as osp
import os
import pickle

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddOrthonormalNodeIdentifiers
from torch_geometric.transforms.compose import Compose
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

import wandb
import networkx as nx

from models.add_substructure_instances import AddSubstructureInstances
from tokengt_experiments.substructs_emb.zinc_substructs_models import GCNGraphRegression, TokenGTGraphRegression


class AddSubstructureEmbeddings(BaseTransform):
    
    def forward(self, data) -> Data:
        for substruct_instances in data.substructure_instances:
            flat = list(chain.from_iterable(substruct_instances))
            counts = torch.bincount(torch.tensor(flat, dtype=torch.long), minlength=data.x.shape[0])
            data.x = torch.cat([data.x, counts.unsqueeze(1)], dim=1)
        
        data.edge_attr = None
        
        return data


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(loader.dataset)


def get_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y.unsqueeze(1)).item()
            total_loss += loss
    return total_loss / len(loader.dataset)

def create_model(config, device, dim_node):
    if config.architecture == "TokenGT":
        if config.use_one_hot_encoding:
            dim_node = 28+dim_node-1 # 28 is the number of different atoms in ZINC
        else:
            dim_node = dim_node
        return TokenGTGraphRegression(
            dim_node=dim_node,
            dim_edge=1,
            d_p=config.D_P,
            d=config.d,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            include_graph_token=config.include_graph_token,
            is_laplacian_node_ids=config.use_laplacian,
            use_one_hot_encoding=config.use_one_hot_encoding,
            dropout=config.dropout,
            device=device,
        )
    elif config.architecture == "GCN":
        return GCNGraphRegression(
            dim_node=dim_node,
            hidden_channels=config.d,
            num_layers=config.num_encoder_layers,
            dropout=config.dropout,
            batch_norm=config.batch_norm,
            use_one_hot_encoding=config.use_one_hot_encoding,
            device=device,
        )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


def load_substructures(filepath: str):
    """Load substructures from pickle file."""
    with open(filepath, 'rb') as f:
        subs = pickle.load(f)
        out = []
        for s in subs:
            G = nx.Graph()
            G.add_edges_from(s)
            out.append(G)
        return out

def main(config):
    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)
    # torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True

    run = wandb.init(
        entity="krecharles-university-of-oxford",
        project="substructure_embeddings",
        config=config,
        # mode="disabled"
    )

    config = wandb.config

    if config.substructures_file == "":
        substructures = []
    else:
        substructures = load_substructures(f"tokengt_experiments/{config.substructures_file}.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        AddOrthonormalNodeIdentifiers(config.D_P, config.use_laplacian),
        AddSubstructureInstances(substructures),
        AddSubstructureEmbeddings()
    ])

    path = osp.join(osp.realpath(os.getcwd()),
                    "data", f"ZINC-{config.use_laplacian}-{config.D_P}-{config.substructures_file}")
    
    train_dataset = ZINC(path, subset=True, split="train",
                         pre_transform=transform)
    val_dataset = ZINC(path, subset=True, split="val", pre_transform=transform)
    test_dataset = ZINC(path, subset=True, split="test", pre_transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    model = create_model(config, device, train_dataset.num_node_features)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of params: {num_params}")
    print(f"n_substructures: {len(substructures)}")
    run.log({"num_param": num_params})

    criterion = nn.L1Loss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.lr_reduce_factor, 
        min_lr=config.min_lr, 
        patience=config.patience,
    )

    train_loss = get_loss(model, train_loader, criterion, device)
    val_loss = get_loss(model, val_loader, criterion, device)
    print(f"Epoch 0: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
    run.log({"train_loss": train_loss, "val_loss": val_loss}, step=1)

    for i in range(2, config.epochs + 2):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = get_loss(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {i}: train_loss={train_loss:.5f} val_loss={val_loss:.5f} lr={current_lr:.6f}")
        run.log({
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "learning_rate": current_lr
        }, step=i)

    test_loss = get_loss(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.5f}")
    run.log({"test_loss": test_loss})

    run.finish()


if __name__ == "__main__":
    # config = {
    #     "architecture": "GCN",  # Options: "TokenGT", "GCN"
    #     "dataset": "ZINC_12K", 
    #     # set substructure_file to "" to use no substructures
    #     "substructures_file": "cycles_3_8",
    #     "D_P": 32,
    #     "num_heads": 8,
    #     "d": 125,
    #     "num_encoder_layers": 4,
    #     "dim_feedforward": 64,
    #     "include_graph_token": True,
    #     "use_laplacian": False,
    #     "use_one_hot_encoding": True,
    #     "batch_norm": True,
    #     "dropout": 0,
    #     "epochs": 250,
    #     "lr": 0.001,
    #     "lr_reduce_factor": 0.5,
    #     "min_lr": 0.00001,
    #     "patience": 10,
    #     "batch_size": 128,
    # }
    config = {
        "architecture": "TokenGT",  # Options: "TokenGT", "GCN"
        "dataset": "ZINC_12K", 
        # set substructure_file to "" to use no substructures
        "substructures_file": "",
        "D_P": 32,
        "num_heads": 8,
        "d": 64,
        "num_encoder_layers": 4,
        "dim_feedforward": 64,
        "include_graph_token": True,
        "use_laplacian": False,
        "use_one_hot_encoding": False,
        "batch_norm": True,
        "dropout": 0.1,
        "epochs": 100,
        "lr": 0.001,
        "lr_reduce_factor": 0.5,
        "min_lr": 0.00001,
        "patience": 10,
        "batch_size": 128,
        "weight_decay": 0.01,
    }
    main(config)
