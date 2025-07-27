from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.utils import data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from models.add_smarts_instances import AddSmartsInstances, AddSubstructureEmbeddings
from tokengt_paper_repo.wrapper import AddTokenGTPaperNodeIdentifiers


class QM9Dataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        d_p: int = 32,
        node_id_mode: str = "orf",
        data_dir: str = "data",
        smarts_patterns: List[List[str]] = [],
        embed_smarts: bool = False,
        target_idx: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.d_p = d_p
        self.node_id_mode = node_id_mode
        self.data_dir = data_dir
        self.smarts_patterns = smarts_patterns
        self.embed_smarts = embed_smarts
        self.target_idx = target_idx

        flatten = lambda lst: [item for sublist in lst for item in sublist]
        self.root_f = f"{self.data_dir}/qm9_{node_id_mode}_{d_p}_{embed_smarts}_{'_'.join(flatten(self.smarts_patterns))}"
        
        self.transform = self.get_transforms()

    def get_transforms(self) -> Compose:
        transforms = [
            FormatQM9Features(self.target_idx),
            AddTokenGTPaperNodeIdentifiers(self.d_p, convert_to_single_emb_offset=10),
        ]
        if len(self.smarts_patterns) > 0:
            transforms.append(AddSmartsInstances(self.smarts_patterns))
            if self.embed_smarts:
                transforms.append(AddSubstructureEmbeddings(len(self.smarts_patterns)))

        return Compose(transforms)


    def setup(self, stage: Optional[str] = None):
        dataset = QM9(root=self.root_f, pre_transform=self.transform)
        dataset = dataset.shuffle()

        self.train, self.val, self.test = data.random_split(
            dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
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

class FormatQM9Features(object):
    def __init__(self, target_idx):
        self.target_idx = target_idx

    def __call__(self, data):
        # only keep atom number and number of hydrogens
        relevant_features = torch.tensor([5, 10])
        data.x = data.x[:, relevant_features]
        data.y = data.y[:, self.target_idx]
        return data