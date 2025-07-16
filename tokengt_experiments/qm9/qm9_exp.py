import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddOrthonormalNodeIdentifiers
from torch_geometric.transforms.compose import Compose
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import wandb
from rdkit import Chem
from typing import List
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tokengt_experiments.qm9.qm9_models import GCNGraphRegression, TokenGTGraphRegression, TokenGTSTSumGraphRegression


class AddSmartsInstances(BaseTransform):

    def __init__(self, smarts_patterns: List[str]):
        self._smarts_patterns = smarts_patterns
        
        # Calculate max atoms across all patterns
        max_atoms = 0
        for s in smarts_patterns:
            mol = Chem.MolFromSmarts(s)
            if mol:
                num_atoms = mol.GetNumAtoms()
                max_atoms = max(max_atoms, num_atoms)
        self._max_atoms = max_atoms

    def forward(self, data: Data) -> Data:
        mol = Chem.MolFromSmiles(data.smiles)
        substructure_instances = []

        if mol is not None:
            for i, smarts in enumerate(self._smarts_patterns):
                pattern = Chem.MolFromSmarts(smarts)
                instances = mol.GetSubstructMatches(pattern)
                instances = [[i] + list(instance) + [-1] * (self._max_atoms - len(instance)) for instance in instances]
                substructure_instances.extend(instances)

        data["substructure_instances"] = torch.tensor(substructure_instances, dtype=torch.long)
        data["n_substructure_instances"] = torch.tensor(len(substructure_instances), dtype=torch.long)
        return data


def train(model, loader, criterion, optimizer, target_idx=0, device=None):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        # QM9 has 19 targets, select one for training
        target = batch.y[:, target_idx].unsqueeze(1)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(loader.dataset)


def get_loss(model, loader, criterion, target_idx=0, device=None) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            target = batch.y[:, target_idx].unsqueeze(1)
            loss = criterion(out, target).item()
            total_loss += loss
    return total_loss / len(loader.dataset)

def get_qm9_smarts_patterns():
    hydrocarbons = [
        "[C]=[C]",           # Alkene
        "c1ccccc1",          # Aromatic ring (benzene)
        "[C]#[C]"            # Alkyne
    ]

    haloalkanes = [
        # "[CX4][F]",          # Fluoroalkane
        # "[CX4][Cl]",         # Chloroalkane
        # "[CX4][I]"           # Iodoalkane
    ]

    oxygen_containing = [
        # "[CX3](=O)[OX2H1]",  # Carboxylic acid
        "[CX3](=O)[#6]",     # Ketone (carbonyl next to carbon)
        "[OD2]([#6])[#6]"    # Ether
    ]

    nitrogen_containing = [
        "[NX3][CX3](=O)[#6]",  # Amide
        "[CX3]=[NX2]",         # Imine
        "[NX3;H2][#6]"         # Primary amine
    ]

    return hydrocarbons + haloalkanes + oxygen_containing + nitrogen_containing


def create_model(config, n_substructures, device):
    if config.architecture == "TokenGT":
        return TokenGTGraphRegression(
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
    elif config.architecture == "TokenGTST_Sum":
        return TokenGTSTSumGraphRegression(
            d_p=config.D_P,
            d=config.d,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            include_graph_token=config.include_graph_token,
            is_laplacian_node_ids=config.use_laplacian,
            dropout=config.dropout,
            device=device,
            n_substructures=n_substructures
        )
    elif config.architecture == "GCN":
        return GCNGraphRegression(
            hidden_channels=config.d,
            num_layers=config.num_encoder_layers,
            dropout=config.dropout,
            batch_norm=True,
            device=device,
        )
    # elif config.architecture == "TokenGTST_Hyp":
    #     return TokenGTSTHypGraphRegression(
    #         dim_node=num_node_features,
    #         d_p=config.D_P,
    #         d=config.d,
    #         num_heads=config.num_heads,
    #         num_encoder_layers=config.num_encoder_layers,
    #         dim_feedforward=config.dim_feedforward,
    #         include_graph_token=config.include_graph_token,
    #         is_laplacian_node_ids=config.use_laplacian,
    #         dim_edge=num_edge_features,
    #         dropout=config.dropout,
    #         device=device,
    #         n_substructures=n_substructures
    #     )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


def main():
    config = {
        "architecture": "GCN",
        "dataset": "QM9",
        "D_P": 32,
        "num_heads": 8,
        "d": 64,
        "num_encoder_layers": 4,
        "dim_feedforward": 64,
        "include_graph_token": True,
        "use_laplacian": False,
        "dropout": 0.1,
        "epochs": 100,
        "lr": 0.001,
        "batch_size": 512,
        "target_idx": 0, # HOMO
    }

    run = wandb.init(
        entity="krecharles-university-of-oxford",
        project="QM9",
        config=config,
        mode="disabled"
    )

    config = wandb.config

    smarts_patterns = get_qm9_smarts_patterns()
    n_substructures = len(smarts_patterns)
    
    transform = Compose([
        AddOrthonormalNodeIdentifiers(config.D_P, config.use_laplacian),
        AddSmartsInstances(smarts_patterns)
    ])

    print("Loading dataset...")
    root_f = f"data/qm9_{'_'.join(smarts_patterns)}"
    dataset = QM9(root=root_f, pre_transform=transform)
    dataset = dataset.shuffle()
    print("Dataset loaded")
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # Convert to list for splitting
    dataset_list = list(dataset)
    train_indices, temp_indices = train_test_split(
        range(len(dataset_list)), train_size=train_size, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=val_size, random_state=42
    )
    
    train_dataset = [dataset_list[i] for i in train_indices]
    val_dataset = [dataset_list[i] for i in val_indices]
    test_dataset = [dataset_list[i] for i in test_indices]
    
    print(f"Training with {len(train_dataset)} samples")
    print(f"Validation with {len(val_dataset)} samples")
    print(f"Testing with {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(config, n_substructures, device)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of params: {num_params}")
    run.log({"num_param": num_params})

    criterion = nn.L1Loss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    for i in range(1, config.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, config.target_idx, device)
        val_loss = get_loss(model, val_loader, criterion, config.target_idx, device)
        print(f"Epoch {i}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        run.log({"train_loss": train_loss, "val_loss": val_loss}, step=i)

    # Final test evaluation
    test_loss = get_loss(model, test_loader, criterion, config.target_idx, device)
    print(f"Test loss: {test_loss:.5f}")
    run.log({"test_loss": test_loss})

    # Save model
    save_path = f"trained_models/{config.architecture}_{config.dataset}_target{config.target_idx}.pt"
    torch.save(model, save_path)

    run.finish()


if __name__ == "__main__":
    main()
