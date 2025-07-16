import pytorch_lightning as pl
from torch.utils import data
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Compose
from torch_geometric.transforms.add_laplacian_node_identifiers import AddLaplacianNodeIdentifiers
from torch_geometric.loader import DataLoader
from typing import Optional
import torch
from .add_smarts_instances import get_qm9_smarts_patterns, AddSmartsInstances

class QM9DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        d_p: int = 32,
        use_laplacian: bool = False,
        data_dir: str = "data",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.d_p = d_p
        self.use_laplacian = use_laplacian
        self.data_dir = data_dir
        self.smarts_patterns = get_qm9_smarts_patterns()
        self.root_f = f"{self.data_dir}/qm9_{'_'.join(self.smarts_patterns)}"
        
        
        if self.use_laplacian:
            self.transform = Compose([
                AddLaplacianNodeIdentifiers(d_p),
                AddSmartsInstances(self.smarts_patterns)
            ])
        else:
            self.transform = Compose([
                AddSmartsInstances(self.smarts_patterns)
            ])

    def prepare_data(self):
        print("++++ Downloading dataset ++++")
        QM9(root=self.root_f)
        print("++++ Dataset downloaded ++++")

    def setup(self, stage: Optional[str] = None):
        dataset = QM9(root=self.root_f, pre_transform=self.transform)
        dataset = dataset.shuffle()

        print("++++ Splitting dataset ++++")

        self.train, self.val, self.test = data.random_split(
            dataset, [0.9, 0.1, 0.0], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            persistent_workers=True
        )
