import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import networkx as nx

from thesis_experiments.zinc_smarts_pattern import get_zinc_smarts_patterns, get_zinc_smarts_patterns_xl
from tokengt_paper_experiments.zinc_dataset import ZincDataset
from thesis_experiments.gcn_model import GCNGraphRegression

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
        )
        num_atoms = 28 + len(smarts_patterns) if config["use_mvn"] else 28
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
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

    parser.add_argument('--architecture', type=str, choices=['GCN'],
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['ZINC'],
                        help='Dataset to use')

    parser.add_argument('--smarts_config', type=str, 
                        choices=['none', 'brics', 'brics_xl'],
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

    parser.add_argument('--hidden_channels', type=int, help='Hidden dimension')
    parser.add_argument('--num_encoder_layers', type=int,
                        help='Number of encoder layers')
    parser.add_argument('--dropout', type=float, help='Dropout rate')

    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_workers', type=int,
                        help='Number of workers for data loading')
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


def get_patterns(config):
    # assert at least one of the configs is not none
    assert config["smarts_config"] != "none" or config["substructures_config"] != "none", "At least one of the configs must be not none"

    if config["smarts_config"] == "brics":
        smarts_patterns = get_zinc_smarts_patterns()
    elif config["smarts_config"] == "brics_xl":
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
    trainer.test(model, data_module)


def main():

    config = {
        "architecture": ["GCN"][0],
        "dataset": ["ZINC"][0],

        # Whether to add the smarts patterns to the node features (and one-hot encode the group index)
        "embed_smarts": True,
        "use_mvn": False,

        "smarts_config": ["none", "brics", "brics_xl"][0],
        "substructures_config": ["none", "cycles"][1],

        "hidden_channels": 16,
        "num_encoder_layers": 2,

        "epochs": 10,
        "batch_size": 64,
        "lr": 0.001,
        "num_workers": 2,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "seed": 1,
    }

    config = parse_arguments(config)

    wandb.init(
        project=f"zinc_thesis",
        entity="krecharles-university-of-oxford",
        config=config,
        mode="disabled"
    )

    train(config)


if __name__ == "__main__":
    main()