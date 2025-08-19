import pytorch_lightning as pl
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from typing import List, Optional
import torch
import networkx as nx

from torch_geometric.transforms.base_transform import BaseTransform

from models.add_smarts_instances import AddSmartsInstances
from models.add_substructure_instances import AddSubstructureInstances
from models.add_substructure_embeddings import AddSubstructureEmbeddings
from models.add_vn_transforms import AddGlobalVN, AddSubstructureMatchesAsVNs, AddSubstructureMatchesAsVNsFullyConnected, AddSubstructureMatchesAsVNsSharingConstituentConnected
from tokengt_paper_experiments.molhiv_pyg_smiles import PygGraphPropPredDatasetWithSmiles

class MolHIVDataset(pl.LightningDataModule):
    
    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        smarts_patterns: List[List[str]] = [],
        substructures_patterns: List[nx.Graph] = [],
        embed_smarts: bool = False,
        use_mvn: bool = False,
        use_mvn_fully_connected: bool = False,
        use_mvn_sharing_connected: bool = False,
        use_global_vn: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.smarts_patterns = smarts_patterns
        self.substructures_patterns = substructures_patterns
        self.embed_smarts = embed_smarts
        self.use_mvn = use_mvn
        self.use_mvn_fully_connected = use_mvn_fully_connected
        self.use_mvn_sharing_connected = use_mvn_sharing_connected
        self.use_global_vn = use_global_vn

        flatten = lambda lst: [item for sublist in lst for item in sublist]
        self.root_f = f"data/molhiv_{embed_smarts}_{len(flatten(self.smarts_patterns))}_{len(self.substructures_patterns)}_{use_mvn}_{use_mvn_fully_connected}_{use_mvn_sharing_connected}_{use_global_vn}"
        
        self.transform = self.get_transforms()


    def get_transforms(self) -> Compose:
        transforms = []
        transforms.append(RemoveOGBFeatures())
        
        if len(self.smarts_patterns) > 0:
            transforms.append(AddSmartsInstances(self.smarts_patterns))
        
        if len(self.substructures_patterns) > 0:
            transforms.append(AddSubstructureInstances(self.substructures_patterns))

        assert not self.embed_smarts or not self.use_mvn, "Cannot embed smarts and use MVN at the same time"

        pattern_len = len(self.smarts_patterns) + len(self.substructures_patterns)
        total_atom_types = 88
        if self.embed_smarts:
            transforms.append(AddSubstructureEmbeddings(pattern_len))
        if self.use_mvn:
            transforms.append(AddSubstructureMatchesAsVNs(pattern_len, total_atom_types))
            total_atom_types += pattern_len
        if self.use_mvn_fully_connected:
            transforms.append(AddSubstructureMatchesAsVNsFullyConnected(pattern_len, total_atom_types))
            total_atom_types += pattern_len
        if self.use_mvn_sharing_connected:
            transforms.append(AddSubstructureMatchesAsVNsSharingConstituentConnected(pattern_len, total_atom_types))
            total_atom_types += pattern_len
        if self.use_global_vn:
            transforms.append(AddGlobalVN(total_atom_types))

        return Compose(transforms)


    def setup(self, stage: Optional[str] = None):

        dataset = PygGraphPropPredDatasetWithSmiles(name="ogbg-molhiv", root = self.root_f, pre_transform=self.transform)

        split_idx = dataset.get_idx_split()

        self.train_loader = DataLoader(
            dataset[split_idx["train"]], 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers = self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        self.valid_loader = DataLoader(
            dataset[split_idx["valid"]], 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers = self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        self.testdev_loader = DataLoader(
            dataset[split_idx["test"]],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
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


class RemoveOGBFeatures(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        data.x = data.x[:, :1]
        return data

