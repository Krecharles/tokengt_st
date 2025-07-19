import pytorch_lightning as pl
from torch.utils import data
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Compose
from torch_geometric.transforms.add_tokengt_node_identifiers import AddLaplacianNodeIdentifiers, AddPrecomputedORFNodeIdentifiers
from torch_geometric.loader import DataLoader
from typing import List, Optional
import torch
from .add_smarts_instances import AddSubstructureEmbeddings, get_qm9_smarts_patterns, AddSmartsInstances

class QM9DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        d_p: int = 32,
        node_id_mode: str = "orf",
        data_dir: str = "data",
        smarts_patterns: List[List[str]] = [],
        embed_smarts: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.d_p = d_p
        self.node_id_mode = node_id_mode
        self.data_dir = data_dir
        self.smarts_patterns = smarts_patterns
        self.embed_smarts = embed_smarts
        
        flatten = lambda lst: [item for sublist in lst for item in sublist]
        self.root_f = f"{self.data_dir}/qm9_{node_id_mode}_{d_p}_{embed_smarts}_{'_'.join(flatten(self.smarts_patterns))}"
        
        self.transform = self.get_transforms()

    def get_transforms(self) -> Compose:
        transforms = []
        if self.node_id_mode == "laplacian":
            transforms.append(AddLaplacianNodeIdentifiers(self.d_p))
        elif self.node_id_mode == "precomputed":
            transforms.append(AddPrecomputedORFNodeIdentifiers(self.d_p))
        
        transforms.append(AddSmartsInstances(self.smarts_patterns))

        if self.embed_smarts:
            transforms.append(AddSubstructureEmbeddings(len(self.smarts_patterns)))

        return Compose(transforms)


    def setup(self, stage: Optional[str] = None):
        dataset = QM9(root=self.root_f, pre_transform=self.transform)
        dataset = dataset.shuffle()

        self.train, self.val, self.test = data.random_split(
            dataset, [0.9, 0.1, 0.0], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True
        )
