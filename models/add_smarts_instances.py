from torch_geometric.transforms import BaseTransform
from typing import List
from rdkit import Chem
from torch_geometric.data import Data
import torch

class AddSmartsInstances(BaseTransform):

    def __init__(self, smarts_patterns: List[List[str]], grouped: bool = False):
        self._smarts_patterns = smarts_patterns
        self._grouped = grouped
        
        max_atoms = 0
        for group in smarts_patterns:
            for s in group:
                mol = Chem.MolFromSmarts(s)
                if mol:
                    num_atoms = mol.GetNumAtoms()
                    max_atoms = max(max_atoms, num_atoms)
        self._max_atoms = max_atoms

    def forward(self, data: Data) -> Data:
        mol = Chem.MolFromSmiles(data.smiles)
        substructure_instances = []

        if mol is not None:
            for group_idx, group in enumerate(self._smarts_patterns):
                for smarts in group:
                    pattern = Chem.MolFromSmiles(smarts)
                    instances = mol.GetSubstructMatches(pattern)
                    instances = [[group_idx] + list(instance) + [-1] * (self._max_atoms - len(instance)) for instance in instances]
                    substructure_instances.extend(instances)
        
        if len(substructure_instances) == 0:
            data["substructure_instances"] = torch.zeros((0, self._max_atoms + 1), dtype=torch.long)
        else:
            data["substructure_instances"] = torch.tensor(substructure_instances, dtype=torch.long)
        data["n_substructure_instances"] = torch.tensor(len(substructure_instances), dtype=torch.long)
        return data

def get_qm9_smarts_patterns():
    mgssl_motifs = [
        'O=CO', # 0
        'C=O', # 1
        'NC=O', # 2
        'C1=CC=CC=C1', # 3
        'NCCO', # 4
        'COC', # 5
        'C=N', # 6  
        'N=CO', # 7
        'C=C', # 8
        'NC1=CC=CC=C1' # 9
    ]
    mgssl_motifs = [ [m] for m in mgssl_motifs ]
    return mgssl_motifs


class AddSubstructureEmbeddings(BaseTransform):
    """
    Augments the node features with by n_substructures where the i-th additional 
    feature is the number of time the given node is a member of the i-th substructure.
    """

    def __init__(self, n_substructures, accumulate: bool = True):
        self.n_substructures = n_substructures # Number of substructures
        self.accumulate = accumulate
    
    def forward(self, data) -> Data:
        if len(data.substructure_instances) == 0:
            emb = torch.zeros(data.x.shape[0], self.n_substructures, dtype=torch.long)
        else:
            keys = data.substructure_instances[:, 0]
            vertices = data.substructure_instances[:, 1:]

            valid_mask = vertices != -1
            flat_vertices = vertices[valid_mask]
            repeated_keys = keys.unsqueeze(1).expand_as(vertices)[valid_mask]

            emb = torch.zeros(data.x.shape[0], self.n_substructures, dtype=torch.long)
            emb.index_put_((flat_vertices, repeated_keys), torch.ones_like(flat_vertices), accumulate=self.accumulate)

        data.x = torch.cat([data.x, emb], dim=1)

        return data