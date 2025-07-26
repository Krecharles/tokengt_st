from torch_geometric.transforms import BaseTransform
from typing import List
from rdkit import Chem
from torch_geometric.data import Data
import torch

class AddSmartsInstances(BaseTransform):

    def __init__(self, smarts_patterns: List[List[str]], grouped: bool = False):
        self._smarts_patterns = smarts_patterns
        self._grouped = grouped
        
        # Calculate max atoms across all patterns
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
                    pattern = Chem.MolFromSmarts(smarts)
                    instances = mol.GetSubstructMatches(pattern)
                    instances = [[group_idx] + list(instance) + [-1] * (self._max_atoms - len(instance)) for instance in instances]
                    substructure_instances.extend(instances)

        data["substructure_instances"] = torch.tensor(substructure_instances, dtype=torch.long)
        data["n_substructure_instances"] = torch.tensor(len(substructure_instances), dtype=torch.long)
        return data

def get_qm9_smarts_patterns(grouped: bool = False):
    hydrocarbons = [
        "[C]=[C]",           # Alkene
        "c1ccccc1",          # Aromatic ring (benzene)
        "[C]#[C]"            # Alkyne
    ]

    haloalkanes = [
        # "[CX4][F]",          # Fluoroalkane
        # "[CX4][Cl]",         # Chloroalkane
        # "[CX4][I]"           # Iodoalkane
    ]

    oxygen_containing = [
        # "[CX3](=O)[OX2H1]",  # Carboxylic acid
        "[CX3](=O)[#6]",     # Ketone (carbonyl next to carbon)
        "[OD2]([#6])[#6]"    # Ether
    ]

    nitrogen_containing = [
        "[NX3][CX3](=O)[#6]",  # Amide
        "[CX3]=[NX2]",         # Imine
        "[NX3;H2][#6]"         # Primary amine
    ]

    if grouped:
        return [hydrocarbons, haloalkanes, oxygen_containing, nitrogen_containing]
    else:
        return [[x] for x in hydrocarbons + haloalkanes + oxygen_containing + nitrogen_containing]

class AddSubstructureEmbeddings(BaseTransform):
    """
    Augments the node features with by n_substructures where the i-th additional 
    feature is the number of time the given node is a member of the i-th substructure.
    """

    def __init__(self, n_substructures):
        self.n_substructures = n_substructures # Number of substructures
    
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
            emb.index_put_((flat_vertices, repeated_keys), torch.ones_like(flat_vertices), accumulate=True)

        data.x = torch.cat([data.x, emb], dim=1)

        return data