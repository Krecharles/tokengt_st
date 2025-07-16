import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from tokengt_experiments.qm9.add_smarts_instances import get_qm9_smarts_patterns
from tokengt_experiments.qm9.qm9_models import GCNGraphRegression, TokenGTGraphRegression, TokenGTSTSumGraphRegression
from tokengt_experiments.qm9.qm9_dataset import QM9DataModule

def create_model(config, n_substructures, device):
    """Create model based on architecture configuration."""
    if config["architecture"] == "TokenGT":
        return TokenGTGraphRegression(
            d_p=config["D_P"],
            d=config["d"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            include_graph_token=config["include_graph_token"],
            is_laplacian_node_ids=config["use_laplacian"],
            dropout=config["dropout"],
            device=device,
            lr=config["lr"],
            target_idx=config["target_idx"],
            batch_size=config["batch_size"],
        )
    elif config["architecture"] == "TokenGTST_Sum":
        return TokenGTSTSumGraphRegression(
            d_p=config["D_P"],
            d=config["d"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            include_graph_token=config["include_graph_token"],
            is_laplacian_node_ids=config["use_laplacian"],
            dropout=config["dropout"],
            device=device,
            n_substructures=n_substructures,
            lr=config["lr"],
            target_idx=config["target_idx"],
            batch_size=config["batch_size"],
        )
    elif config["architecture"] == "GCN":
        return GCNGraphRegression(
            hidden_channels=config["d"],
            num_layers=config["num_encoder_layers"],
            dropout=config["dropout"],
            batch_norm=True,
            device=device,
            lr=config["lr"],
            target_idx=config["target_idx"],
            batch_size=config["batch_size"],
        )
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")


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
        "batch_size": 32,
        "target_idx": 2,  # HOMO
        "num_workers": 4,
    }

    data_module = QM9DataModule(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        d_p=config["D_P"],
        use_laplacian=config["use_laplacian"],
    )

    n_substructures = len(get_qm9_smarts_patterns())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config, n_substructures, device)

    wandb.init(mode="disabled")

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        logger=WandbLogger(
            project="QM9_temp",
            entity="krecharles-university-of-oxford",
            log_model=True,
        ),
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    # Save trained model
    save_path = f"trained_models/{config['architecture']}_{config['dataset']}_target{config['target_idx']}.pt"
    torch.save(model, save_path)


if __name__ == "__main__":
    main() 