import pytorch_lightning as pl
from torch.utils import data
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from typing import Optional

from tokengt_experiments.model_analysis.motif_selection.zinc_smiles_dataset import ZincSmilesDataset
from tokengt_paper_repo.wrapper import AddTokenGTPaperNodeIdentifiers
from tokengt_pyg.add_laplacian_node_ids import AddLaplacianNodeIdentifiers

class ZincDataset(pl.LightningDataModule):
    SINGLE_EMB_OFFSET = 28

    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        d_p: int = 32,
        node_id_mode: str = "orf",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.d_p = d_p
        self.node_id_mode = node_id_mode

        self.root_f = f"data/zinc_{self.d_p}_{self.node_id_mode}"
        
        self.transform = self.get_transforms()

    def get_transforms(self) -> Compose:
        transforms = []
        if self.node_id_mode == "laplacian":
            transforms.append(AddLaplacianNodeIdentifiers(self.d_p))

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
