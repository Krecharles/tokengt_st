import pytorch_lightning as pl
from torch.utils import data
from torch_geometric.datasets import ZINC
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from typing import List, Optional

from tokengt_paper_repo.wrapper import AddTokenGTPaperNodeIdentifiers
from models.add_smarts_instances import AddSubstructureEmbeddings, AddSmartsInstances

class ZincDataset(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        d_p: int = 32,
        node_id_mode: str = "orf",
        smarts_patterns: List[List[str]] = [],
        embed_smarts: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.d_p = d_p
        self.node_id_mode = node_id_mode
        self.smarts_patterns = smarts_patterns
        self.embed_smarts = embed_smarts
        
        flatten = lambda lst: [item for sublist in lst for item in sublist]
        self.root_f = f"data/zinc_{d_p}_{embed_smarts}_{'_'.join(flatten(self.smarts_patterns))}"
        
        self.transform = self.get_transforms()

    def get_transforms(self) -> Compose:
        transforms = [
            AddTokenGTPaperNodeIdentifiers(self.d_p),
        ]
        if len(self.smarts_patterns) > 0:
            transforms.append(AddSmartsInstances(self.smarts_patterns))
            if self.embed_smarts:
                transforms.append(AddSubstructureEmbeddings(len(self.smarts_patterns)))

        return Compose(transforms)


    def setup(self, stage: Optional[str] = None):
        self.train = ZINC(root=self.root_f, subset=True, pre_transform=self.transform, split="train")
        self.val = ZINC(root=self.root_f, subset=True, pre_transform=self.transform, split="val")
        self.test = ZINC(root=self.root_f, subset=True, pre_transform=self.transform, split="test")

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
