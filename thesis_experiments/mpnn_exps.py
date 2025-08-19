import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import networkx as nx
import numpy as np

from thesis_experiments.zinc_smarts_pattern import get_zinc_smarts_patterns, get_zinc_smarts_patterns_xl
from tokengt_paper_experiments.molhiv_dataset import MolHIVDataset
from tokengt_paper_experiments.zinc_dataset import ZincDataset
from thesis_experiments.mpnn_models import GCNGraphRegression, GATGraphRegression

def create_model(config, num_atoms, n_substructures):
    if config["architecture"] == "GCN":
        return GCNGraphRegression(
            num_node_features=num_atoms,
            num_substructures=n_substructures if config["embed_smarts"] else 0,
            hidden_channels=config["hidden_channels"],
            num_layers=config["num_encoder_layers"],
            dropout=config["dropout"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            reduce_factor=config["reduce_factor"],
            stopping_learning_rate=config["stopping_learning_rate"],
            patience=config["patience"],
        )
    elif config["architecture"] == "GAT":
        return GATGraphRegression(
            num_node_features=num_atoms,
            num_substructures=n_substructures if config["embed_smarts"] else 0,
            hidden_channels=config["hidden_channels"],
            num_layers=config["num_encoder_layers"],
            heads=config["heads"],
            dropout=config["dropout"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            reduce_factor=config["reduce_factor"],
            stopping_learning_rate=config["stopping_learning_rate"],
            patience=config["patience"],
        )
    else:
        raise ValueError(f"Unknown architecture: {config['architecture']}")


def get_data_module_and_sizes(config, smarts_patterns, substructures_patterns):
    if config["dataset"] == "ZINC":
        data_module = ZincDataset(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            smarts_patterns=smarts_patterns,
            substructures_patterns=substructures_patterns,
            embed_smarts=config["embed_smarts"],
            use_mvn=config["use_mvn"],
            use_mvn_fully_connected=config["use_mvn_fully_connected"],
            use_mvn_sharing_connected=config["use_mvn_sharing_connected"],
            use_global_vn=config["use_global_vn"],
        )
        num_atoms = 28
    elif config["dataset"] == "MolHIV":
        data_module = MolHIVDataset(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            smarts_patterns=smarts_patterns,
            substructures_patterns=substructures_patterns,
            embed_smarts=config["embed_smarts"],
            use_mvn=config["use_mvn"],
            use_mvn_fully_connected=config["use_mvn_fully_connected"],
            use_mvn_sharing_connected=config["use_mvn_sharing_connected"],
            use_global_vn=config["use_global_vn"],
        )
        num_atoms = 88  
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    if config["use_mvn"]:
        num_atoms += len(smarts_patterns)
    if config["use_mvn_fully_connected"]:
        num_atoms += len(smarts_patterns)
    if config["use_mvn_sharing_connected"]:
        num_atoms += len(smarts_patterns)
    if config["use_global_vn"]:
        num_atoms += 1
    return data_module, num_atoms

def load_substructures(filepath: str):
    with open(filepath, 'rb') as f:
        subs = pickle.load(f)
        out = []
        for s in subs:
            G = nx.Graph()
            G.add_edges_from(s)
            out.append(G)
        return out

def parse_arguments(config):
    parser = argparse.ArgumentParser(description='MPNN Experiments')

    parser.add_argument('--architecture', type=str, choices=['GCN', 'GAT'],
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['ZINC', 'MolHIV'],
                        help='Dataset to use')

    parser.add_argument('--smarts_config', type=str, 
                        choices=['none', 'smarts', 'smarts-xl'],
                        help='SMARTS pattern configuration')
    parser.add_argument('--substructures_config', type=str,
                        choices=['none', 'cycles'],
                        help='Substructures pattern configuration')

    parser.add_argument('--embed_smarts_yes',
                        action='store_true', help='Enable SMARTS embedding')
    parser.add_argument('--embed_smarts_no',
                        action='store_true', help='Disable SMARTS embedding')
    
    parser.add_argument('--use_mvn_yes',
                        action='store_true', help='Enable MVN')
    parser.add_argument('--use_mvn_no',
                        action='store_true', help='Disable MVN')
    parser.add_argument('--use_mvn_fully_connected_yes', action='store_true')
    parser.add_argument('--use_mvn_fully_connected_no', action='store_true')
    parser.add_argument('--use_mvn_sharing_connected_yes', action='store_true')
    parser.add_argument('--use_mvn_sharing_connected_no', action='store_true')
    parser.add_argument('--use_global_vn_yes', action='store_true')
    parser.add_argument('--use_global_vn_no', action='store_true')

    parser.add_argument('--hidden_channels', type=int, help='Hidden dimension')
    parser.add_argument('--num_encoder_layers', type=int,
                        help='Number of encoder layers')
    parser.add_argument('--heads', type=int, help='Number of heads')
    parser.add_argument('--dropout', type=float, help='Dropout rate')

    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_workers', type=int,
                        help='Number of workers for data loading')
    parser.add_argument('--reduce_factor', type=float, help='Learning rate reduction factor')
    parser.add_argument('--stopping_learning_rate', type=float, help='Minimum learning rate before stopping')
    parser.add_argument('--patience', type=int, help='Patience for learning rate reduction')
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


def get_patterns(config):
    # assert at least one of the configs is not none
    assert config["smarts_config"] == "none" or config["substructures_config"] == "none", "At least one of the configs must be not none"

    if config["smarts_config"] == "smarts":
        smarts_patterns = get_zinc_smarts_patterns()
    elif config["smarts_config"] == "smarts-xl":
        smarts_patterns = get_zinc_smarts_patterns_xl()
    elif config["smarts_config"] == "none":
        smarts_patterns = []
    else:
        raise ValueError(f"Unknown pattern config: {config['smarts_config']}")

    if config["substructures_config"] == "cycles":
        substructures_patterns = load_substructures("thesis_experiments/cycles_3_8.pkl")
    elif config["substructures_config"] == "none":
        substructures_patterns = []
    else:
        raise ValueError(f"Unknown pattern config: {config['substructures_config']}")

    return smarts_patterns, substructures_patterns

def train(config):

    pl.seed_everything(config["seed"], workers=True)

    smarts_patterns, substructures_patterns = get_patterns(config)

    n_substructures = len(smarts_patterns) + len(substructures_patterns)

    data_module, num_atoms = get_data_module_and_sizes(config, smarts_patterns, substructures_patterns)

    wandb_logger = WandbLogger()

    model = create_model(config, num_atoms, n_substructures)
  
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    wandb_logger.experiment.log({"total_parameters": total_params})

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        precision="16-mixed",
        # gradient_clip_val=5.0,
        default_root_dir=f"./",
        # val_check_interval=0.2,
        # check_val_every_n_epoch=2,
        # fast_dev_run=True,
    )

    trainer.fit(model, data_module)
    results = trainer.test(model, data_module)
    print(results)
    return results[0]["test_loss"]


def main():

    config = {
        "architecture": ["GCN", "GAT"][0],
        "dataset": ["ZINC", "MolHIV"][1],

        "embed_smarts": False,
        "use_mvn": False,
        "use_mvn_fully_connected": True,
        "use_mvn_sharing_connected": False,
        "use_global_vn": False,

        "smarts_config": ["none", "smarts", "smarts-xl"][1],
        "substructures_config": ["none", "cycles"][0],

        "hidden_channels": 145,
        "num_encoder_layers": 4,
        "heads": 8,

        "epochs": 200,
        "batch_size": 128,
        "num_workers": 4,
        "dropout": 0.0,
        "seed": 1,
        "lr": 0.001,
        "reduce_factor": 0.5,
        "stopping_learning_rate": 1e-5,
        "patience": 10,
    }

    config = parse_arguments(config)

    results = []
    for i in range(1):
        wandb.init(
            project=f"zinc_thesis",
            entity="krecharles-university-of-oxford",
            config=config,
            mode="disabled"
        )
        config["seed"] = i
        loss = train(config)
        results.append(loss)
        
    wandb.log({"results": results})
    print("-"*25)
    print(results)
    mean_loss = np.mean(results)
    std_error = np.std(results, ddof=1) / np.sqrt(len(results))
    print(f"Mean loss: {mean_loss:.3f}, Std error: {std_error:.3f}")
    print("-"*25)


if __name__ == "__main__":
    main()