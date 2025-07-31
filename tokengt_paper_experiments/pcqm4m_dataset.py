import pytorch_lightning as pl
from torch.utils import data
from torch_geometric.datasets import PCQM4Mv2
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from typing import List, Optional
import torch

from ogb.utils import smiles2graph
from torch_geometric.data import Data

from tokengt_paper_repo.wrapper import AddTokenGTPaperNodeIdentifiers
from models.add_smarts_instances import AddSubstructureEmbeddings, AddSmartsInstances

def ogb_from_smiles_wrapper(smiles, *args, **kwargs):
    """Returns `torch_geometric.data.Data` object from smiles while
    `ogb.utils.smiles2graph` returns a dict of np arrays.
    """
    data_dict = smiles2graph(smiles, *args, **kwargs)
    return Data(
        x=torch.from_numpy(data_dict['node_feat']),
        edge_index=torch.from_numpy(data_dict['edge_index']),
        edge_attr=torch.from_numpy(data_dict['edge_feat']),
        smiles=smiles,
    )

class PCQM4MDataset(pl.LightningDataModule):
    SINGLE_EMB_OFFSET = 512
    
    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        d_p: int = 32,
        node_id_mode: str = "orf",
        smarts_patterns: List[List[str]] = [],
        embed_smarts: bool = False,
        dataset_fraction: float = 1.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.d_p = d_p
        self.node_id_mode = node_id_mode
        self.smarts_patterns = smarts_patterns
        self.embed_smarts = embed_smarts
        self.dataset_fraction = dataset_fraction
        
        flatten = lambda lst: [item for sublist in lst for item in sublist]
        self.root_f = f"data/pcqm4m_{d_p}_{embed_smarts}_{'_'.join(flatten(self.smarts_patterns))}"
        
        self.transform = self.get_transforms()

    def get_transforms(self) -> Compose:
        transforms = []
        if len(self.smarts_patterns) > 0:
            transforms.append(AddSmartsInstances(self.smarts_patterns))
            if self.embed_smarts:
                transforms.append(AddSubstructureEmbeddings(len(self.smarts_patterns)))

        transforms.append(AddTokenGTPaperNodeIdentifiers(self.d_p, convert_to_single_emb_offset=self.SINGLE_EMB_OFFSET))

        return Compose(transforms)

    def setup(self, stage: Optional[str] = None):
        self.train = PCQM4Mv2(
            root=self.root_f, 
            split="train",
            from_smiles=ogb_from_smiles_wrapper,
            transform=self.transform
        )
        self.val = PCQM4Mv2(
            root=self.root_f, 
            split="val",
            from_smiles=ogb_from_smiles_wrapper,
            transform=self.transform
        )
        self.test = PCQM4Mv2(
            root=self.root_f, 
            split="test",
            from_smiles=ogb_from_smiles_wrapper,
            transform=self.transform
        )

        if self.dataset_fraction < 1.0:
            self.train = self.train[:int(len(self.train) * self.dataset_fraction)]
            self.val = self.val[:int(len(self.val) * self.dataset_fraction)]
            # self.test = self.test.shuffle()[:int(len(self.test) * self.dataset_fraction)]
            print(f"Using {self.dataset_fraction*100}% of dataset: train={len(self.train)}, val={len(self.val)}, test={len(self.test)}")

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
