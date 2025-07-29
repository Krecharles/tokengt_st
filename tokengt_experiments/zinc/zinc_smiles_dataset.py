from typing import List, Optional

import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class ZincSmilesDataset(InMemoryDataset):

    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(
        self,
        root: str,
        subset: bool = False,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self) -> str:
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self) -> None:
        fs.rm(self.raw_dir)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

        download_url('https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/refs/heads/master/data/zinc/train.txt', self.raw_dir, filename='smiles_train.txt')
        download_url('https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/refs/heads/master/data/zinc/valid.txt', self.raw_dir, filename='smiles_val.txt')
        download_url('https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/refs/heads/master/data/zinc/test.txt', self.raw_dir, filename='smiles_test.txt')

    def process(self) -> None:

        for split in ['train', 'val', 'test']:
            print(f"Processing {split} split")
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)
                print(f"Loaded {len(mols)} molecules for {split} split")

            with open(osp.join(self.raw_dir, f'smiles_{split}.txt'), 'r') as f:
                smiles = f.readlines()

            print(f"Loaded {len(smiles)} smiles")

            indices = list(range(len(mols)))

            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index')) as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                mol = mols[idx]
                smile = smiles[idx]

                x = mol['atom_type'].to(torch.long).view(-1, 1)
                y = mol['logP_SA_cycle_normalized'].to(torch.float)

                adj = mol['bond_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y, smiles=smile)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            self.save(data_list, osp.join(self.processed_dir, f'{split}.pt'))
