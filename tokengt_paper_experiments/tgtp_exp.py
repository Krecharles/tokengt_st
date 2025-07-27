import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
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
            n_substructures=n_substructures
        )
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")


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

def main():
    config = {
        "architecture": ["TokenGT_Paper", "TokenGT_Paper_Sum"][1],
        "dataset": ["ZINC", "QM9", "PCQM4M"][1],
        "target_idx": 2,  # HOMO
        "embed_smarts": False, # Whether to add the smarts patterns to the node features (and one-hot encode the group index)
        
        "node_id_mode": ["orf", "laplacian"][1],
        "D_P": 64,
        "num_heads": 8,
        "d": 64,
        "num_encoder_layers": 3,

        "epochs": 50,
        "batch_size": 8,
        "lr": 0.0002,
        "num_workers": 4,
        "weight_decay": 0.1,
        "dropout": 0.1,
    }

    model_name = f"{config['architecture']}_{config['dataset']}_target{config['target_idx']}"

    pl.seed_everything(42, workers=True)

    smarts_patterns = get_qm9_smarts_patterns()
    # smarts_patterns = []
    n_substructures = len(smarts_patterns)

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
        num_atoms = 10 * 2
        num_edges = 10 * 4
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
    model = create_model(config, num_atoms, num_edges, n_substructures)

    wandb.init(
        project=f"tgtp_{config['dataset']}",
        entity="krecharles-university-of-oxford",
        config=config,
        mode="disabled"
    )
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
        callbacks=[checkpoint_callback]
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    wandb_logger.experiment.log({"total_parameters": total_params})

    checkpoint_path = load_checkpoint_path(model_name)

    trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    trainer.test(model, data_module)

    # Save trained model
    os.makedirs(f"trained_models/{model_name}", exist_ok=True)
    save_path = f"trained_models/{model_name}/model.pt"
    torch.save(model, save_path)


if __name__ == "__main__":
    main() 