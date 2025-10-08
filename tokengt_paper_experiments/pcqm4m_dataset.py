import pytorch_lightning as pl
from torch.utils import data
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from typing import List, Optional
import torch

from ogb.utils import smiles2graph
from ogb.lsc import PCQM4Mv2Evaluator
from tokengt_paper_repo.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from torch_geometric.data import Data

from tokengt_pyg.add_laplacian_node_ids import ConvertToSingleEmbTransform

def ogb_from_smiles_wrapper(smiles, *args, **kwargs):
    """Returns `torch_geometric.data.Data` object from smiles while
    `ogb.utils.smiles2graph` returns a dict of np arrays.
    """
    data_dict = smiles2graph(smiles, *args, **kwargs)
    return Data(
        x=data_dict['node_feat'],
        edge_index=data_dict['edge_index'],
        edge_attr=data_dict['edge_feat'],
        num_nodes=data_dict['num_nodes'],
        smiles=smiles,
    )

class PCQM4MDataset(pl.LightningDataModule):
    SINGLE_EMB_OFFSET = 64
    
    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        d_p: int = 32,
        node_id_mode: str = "orf",
        smarts_patterns: List[List[str]] = [],
        embed_smarts: bool = False,
        dataset_fraction: float = 1.0,
        add_pe: bool = False,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.d_p = d_p
        self.node_id_mode = node_id_mode
        self.smarts_patterns = smarts_patterns
        self.embed_smarts = embed_smarts
        self.dataset_fraction = dataset_fraction
        self.add_pe = add_pe
        self.prefetch_factor = prefetch_factor
        self.evaluator = PCQM4Mv2Evaluator()

    def get_transforms(self) -> Compose:
        transforms = []
        transforms.append(ConvertToSingleEmbTransform(offset=self.SINGLE_EMB_OFFSET))
        if self.node_id_mode == "laplacian":
            transforms.append(AddLaplacianNodeIdentifiers(self.d_p))

        return Compose(transforms)

    def setup(self, stage: Optional[str] = None):

        dataset = PygPCQM4Mv2Dataset(root = 'data/pcqm4m/', transform=self.get_transforms())
        split_idx = dataset.get_idx_split()

        if self.dataset_fraction < 1.0:
            subset_len = int(self.dataset_fraction*len(split_idx["train"]))
            subset_idx = torch.randperm(len(split_idx["train"]))[:subset_len]
            self.train_loader = DataLoader(
                dataset[split_idx["train"][subset_idx]], 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers = self.num_workers,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
            )
        else:
            self.train_loader = DataLoader(
                dataset[split_idx["train"]], 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers = self.num_workers,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
            )
        self.valid_loader = DataLoader(
            dataset[split_idx["valid"]], 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers = self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )
        self.testdev_loader = DataLoader(
            dataset[split_idx["test-dev"]],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.testdev_loader
    
    @torch.no_grad()
    def evaluate_valid_mae(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        model.eval()
        if device is None:
            device = next(model.parameters()).device

        y_true, y_pred = [], []
        for batch in self.valid_loader:
            batch = batch.to(device)
            pred, _ = model(batch)
            y_true.append(batch.y.view_as(pred).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        out = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        return float(out["mae"])
