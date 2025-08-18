import pytorch_lightning as pl
from torch.utils import data
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from typing import List, Optional
import networkx as nx

from models.add_smarts_instances import AddSmartsInstances
from models.add_substructure_embeddings import AddSubstructureEmbeddings
from models.add_vn_transforms import AddGlobalVN, AddSubstructureMatchesAsVNs, AddSubstructureMatchesAsVNsFullyConnected, AddSubstructureMatchesAsVNsSharingConstituentConnected
from models.add_substructure_instances import AddSubstructureInstances
from tokengt_experiments.motif_selection.zinc_smiles_dataset import ZincSmilesDataset

class ZincDataset(pl.LightningDataModule):

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
        self.root_f = f"data/zinc_{embed_smarts}_{len(flatten(self.smarts_patterns))}_{len(self.substructures_patterns)}_{use_mvn}_{use_mvn_fully_connected}_{use_mvn_sharing_connected}_{use_global_vn}"
        
        self.transform = self.get_transforms()

    def get_transforms(self) -> Compose:
        transforms = []
        if len(self.smarts_patterns) > 0:
            transforms.append(AddSmartsInstances(self.smarts_patterns))
        
        if len(self.substructures_patterns) > 0:
            transforms.append(AddSubstructureInstances(self.substructures_patterns))

        assert not self.embed_smarts or not self.use_mvn, "Cannot embed smarts and use MVN at the same time"

        pattern_len = len(self.smarts_patterns) + len(self.substructures_patterns)
        total_atom_types = 28
        if self.embed_smarts:
            transforms.append(AddSubstructureEmbeddings(pattern_len))
            total_atom_types += pattern_len
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
        self.train = ZincSmilesDataset(root=self.root_f, subset=True, pre_transform=self.transform, split="train")
        self.val = ZincSmilesDataset(root=self.root_f, subset=True, pre_transform=self.transform, split="val")
        self.test = ZincSmilesDataset(root=self.root_f, subset=True, pre_transform=self.transform, split="test")

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
