import pytorch_lightning as pl
from torch.utils import data
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from typing import List, Optional

from tokengt_paper_repo.wrapper import AddTokenGTPaperNodeIdentifiers
from models.add_smarts_instances import AddSubstructureEmbeddings, AddSmartsInstances
from tokengt_experiments.zinc.zinc_smiles_dataset import ZincSmilesDataset

class ZincDataset(pl.LightningDataModule):

    SINGLE_EMB_OFFSET = 30

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
        transforms = []
        if len(self.smarts_patterns) > 0:
            transforms.append(AddSmartsInstances(self.smarts_patterns))
            if self.embed_smarts:
                transforms.append(AddSubstructureEmbeddings(len(self.smarts_patterns)))
        # Add this at the end because we need to add the right offsets to the substructure embeddings.
        transforms.append(AddTokenGTPaperNodeIdentifiers(self.d_p, convert_to_single_emb_offset=self.SINGLE_EMB_OFFSET))

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
