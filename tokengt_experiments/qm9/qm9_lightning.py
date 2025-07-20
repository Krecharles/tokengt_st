import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from tokengt_experiments.qm9.add_smarts_instances import get_qm9_smarts_patterns
from tokengt_experiments.qm9.qm9_models import GCNGraphRegression, TokenGTGraphRegression, TokenGTSTSumGraphRegression, MPNNGraphRegression
from tokengt_experiments.qm9.qm9_dataset import QM9DataModule

def create_model(config, n_substructures):
    """Create model based on architecture configuration."""
    dim_node = 11 + n_substructures if config["embed_smarts"] else 11
    dim_edge = 4
    if config["architecture"] == "TokenGT":
        return TokenGTGraphRegression(
            d_p=config["D_P"],
            dim_node=dim_node,
            dim_edge=dim_edge,
            d=config["d"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            include_graph_token=config["include_graph_token"],
            node_id_mode=config["node_id_mode"],
            dropout=config["dropout"],
            lr=config["lr"],
            target_idx=config["target_idx"],
            batch_size=config["batch_size"],
        )
    elif config["architecture"] == "TokenGTST_Sum":
        return TokenGTSTSumGraphRegression(
            d_p=config["D_P"],
            dim_node=dim_node,
            dim_edge=dim_edge,
            d=config["d"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_encoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            include_graph_token=config["include_graph_token"],
            node_id_mode=config["node_id_mode"],
            dropout=config["dropout"],
            n_substructures=n_substructures,
            lr=config["lr"],
            target_idx=config["target_idx"],
            batch_size=config["batch_size"],
        )
    elif config["architecture"] == "GCN":
        return GCNGraphRegression(
            dim_node=dim_node,
            hidden_channels=config["d"],
            num_layers=config["num_encoder_layers"],
            dropout=config["dropout"],
            batch_norm=True,
            lr=config["lr"],
            target_idx=config["target_idx"],
            batch_size=config["batch_size"],
        )
    elif config["architecture"] == "MPNN":
        return MPNNGraphRegression(
            dim_node=dim_node,
            dim_edge=dim_edge,
            hidden_channels=config["d"],
            num_layers=config["num_encoder_layers"],
            dropout=config["dropout"],
            batch_norm=True,
            lr=config["lr"],
            target_idx=config["target_idx"],
            batch_size=config["batch_size"],
        )
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")


def main():
    config = {
        "architecture": "TokenGTST_Sum",
        "dataset": "QM9",
        "target_idx": 2,  # HOMO
        "D_P": 64,
        "num_heads": 16,
        "d": 128,
        "num_encoder_layers": 8,
        "dim_feedforward": 128,
        "include_graph_token": True,
        "node_id_mode": "orf",
        "dropout": 0,
        "epochs": 100,
        "lr": 0.001,
        "batch_size": 1024,
        "num_workers": 8,
        "group_smarts": True,
        "embed_smarts": False, # Whether to add the smarts patterns to the node features (and one-hot encode the group index)
    }

    pl.seed_everything(42, workers=True)

    smarts_patterns = get_qm9_smarts_patterns(grouped=config["group_smarts"])

    data_module = QM9DataModule(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        d_p=config["D_P"],
        node_id_mode=config["node_id_mode"],
        smarts_patterns=smarts_patterns,
        embed_smarts=config["embed_smarts"],
    )

    n_substructures = len(smarts_patterns)
    
    model = create_model(config, n_substructures)

    wandb.init(
        project="QM9",
        entity="krecharles-university-of-oxford",
        config=config,
        # mode="disabled"
    )
    wandb_logger = WandbLogger()



    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        precision="16-mixed",
        gradient_clip_val=1.0
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    wandb_logger.experiment.log({"total_parameters": total_params})


    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    # Save trained model
    save_path = f"trained_models/{config['architecture']}_{config['dataset']}_target{config['target_idx']}.pt"
    torch.save(model, save_path)


if __name__ == "__main__":
    main() 