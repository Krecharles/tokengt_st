import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint

from models.add_smarts_instances import get_qm9_smarts_patterns
from tokengt_experiments.qm9.qm9_dataset import QM9Dataset
from tokengt_paper_experiments.pcqm4m_dataset import PCQM4MDataset
from tokengt_paper_experiments.zinc_dataset import ZincDataset
from tokengt_paper_repo.tokengt_paper import TokenGTPaperGraphRegression

def create_model(config, num_atoms, num_edges, n_substructures):
    if config["architecture"] == "TokenGT_Paper":
        return TokenGTPaperGraphRegression(
            num_atoms=num_atoms,
            num_edges=num_edges,
            d_p=config["D_P"],
            d=config["d"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_encoder_layers"],
            node_id_mode=config["node_id_mode"],
            dropout=config["dropout"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            weight_decay=config["weight_decay"],
            return_attention=True,
        )
    elif config["architecture"] == "TokenGT_Paper_Sum":
        return TokenGTPaperGraphRegression(
            num_atoms=num_atoms,
            num_edges=num_edges,
            d_p=config["D_P"],
            d=config["d"],
            num_heads=config["num_heads"],
            num_encoder_layers=config["num_encoder_layers"],
            node_id_mode=config["node_id_mode"],
            dropout=config["dropout"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            weight_decay=config["weight_decay"],
            substructure_mode="sum",
            n_substructures=n_substructures,
            return_attention=True,
        )
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")

def get_data_module_and_sizes(config, smarts_patterns, n_substructures):
    if config["dataset"] == "QM9":
        data_module = QM9Dataset(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            d_p=config["D_P"],
            node_id_mode=config["node_id_mode"],
            smarts_patterns=smarts_patterns,
            embed_smarts=config["embed_smarts"],
            target_idx=config["target_idx"],
        )
        num_atoms = 100 * 9
        num_edges = 100 * 4
    elif config["dataset"] == "ZINC":
        data_module = ZincDataset(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            d_p=config["D_P"],
            node_id_mode=config["node_id_mode"],
            smarts_patterns=smarts_patterns,
            embed_smarts=config["embed_smarts"],
        )
        num_atoms = 28 + n_substructures if config["embed_smarts"] else 28
        num_edges = 5
    elif config["dataset"] == "PCQM4M":
        data_module = PCQM4MDataset(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            d_p=config["D_P"],
            node_id_mode=config["node_id_mode"],
        )
        num_atoms = 512 * 9
        num_edges = 512 * 4
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    return data_module, num_atoms, num_edges

def load_checkpoint_path(model_name):
    if os.path.exists(f"./checkpoints/{model_name}"):
        if len(os.listdir(f"./checkpoints/{model_name}")) == 0:
            print("No checkpoint found, training from scratch")
            return None
        elif len(os.listdir(f"./checkpoints/{model_name}")) == 1:
            return os.path.join(f"./checkpoints/{model_name}", os.listdir(f"./checkpoints/{model_name}")[0])
        else:
            print("Multiple checkpoints found, please delete one")
            exit()

def parse_arguments(config):
    parser = argparse.ArgumentParser(description='TokenGT Paper Experiments')
    
    parser.add_argument('--architecture', type=str, choices=['TokenGT_Paper', 'TokenGT_Paper_Sum'], 
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['ZINC', 'QM9', 'PCQM4M'], 
                       help='Dataset to use')
    parser.add_argument('--target_idx', type=int, help='Target index for QM9 dataset')
    
    parser.add_argument('--embed_smarts_yes', action='store_true', help='Enable SMARTS embedding')
    parser.add_argument('--embed_smarts_no', action='store_true', help='Disable SMARTS embedding')
    parser.add_argument('--checkpointing_yes', action='store_true', help='Enable checkpointing')
    parser.add_argument('--checkpointing_no', action='store_true', help='Disable checkpointing')
    
    parser.add_argument('--node_id_mode', type=str, choices=['orf', 'laplacian'], help='Node ID mode')
    parser.add_argument('--D_P', type=int, help='Positional encoding dimension')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--d', type=int, help='Hidden dimension')
    parser.add_argument('--num_encoder_layers', type=int, help='Number of encoder layers')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    for key, value in vars(args).items():
        # handle boolean flags
        if key.endswith("_yes") and value:
            config[key[:-4]] = True
        elif key.endswith("_no") and value:
            config[key[:-3]] = False
        elif value is not None:
            config[key] = value
    
    print(config)
    return config

def train(config):
    model_name = f"{config['architecture']}_{config['dataset']}_target{config['target_idx']}"

    pl.seed_everything(config["seed"], workers=True)

    smarts_patterns = get_qm9_smarts_patterns()
    n_substructures = len(smarts_patterns)

    data_module, num_atoms, num_edges = get_data_module_and_sizes(config, smarts_patterns, n_substructures)
    model = create_model(config, num_atoms, num_edges, n_substructures)

    wandb_logger = WandbLogger()

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{model_name}",
        monitor="val_loss",
        mode="min",     
        save_top_k=1,   
        filename="{epoch}-{val_loss:.4f}",
        save_last=False
    )

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        precision="16-mixed",
        gradient_clip_val=5.0,
        default_root_dir=f"./",
        callbacks=[checkpoint_callback] if config["checkpointing"] else None
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    wandb_logger.experiment.log({"total_parameters": total_params})

    checkpoint_path = load_checkpoint_path(model_name)

    trainer.fit(model, data_module, ckpt_path=checkpoint_path if config["checkpointing"] else None)
    trainer.test(model, data_module)

    # Save trained model
    os.makedirs(f"trained_models/{model_name}", exist_ok=True)
    save_path = f"trained_models/{model_name}/model.pt"
    torch.save(model, save_path)

def main():

    config = {
        "architecture": ["TokenGT_Paper", "TokenGT_Paper_Sum"][0],
        "dataset": ["ZINC", "QM9", "PCQM4M"][0],
        "target_idx": 2,  # HOMO
        "embed_smarts": False, # Whether to add the smarts patterns to the node features (and one-hot encode the group index)
        
        "node_id_mode": ["orf", "laplacian"][1],
        "D_P": 64,
        "num_heads": 16,
        "d": 64,
        "num_encoder_layers": 4,

        "epochs": 50,
        "batch_size": 128,
        "lr": 0.001,
        "num_workers": 8,
        "weight_decay": 0.1,
        "dropout": 0.1,
        "checkpointing": False,
        "seed": 1,
    }

    config = parse_arguments(config)

    wandb.init(
        project=f"tgtp_{config['dataset']}",
        entity="krecharles-university-of-oxford",
        config=config,
        mode="disabled"
    )

    train(config)


if __name__ == "__main__":
    main() 