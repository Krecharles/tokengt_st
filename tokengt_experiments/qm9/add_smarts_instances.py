from torch_geometric.transforms import BaseTransform
from typing import List
from rdkit import Chem
from torch_geometric.data import Data
import torch

class AddSmartsInstances(BaseTransform):

    def __init__(self, smarts_patterns: List[str]):
        self._smarts_patterns = smarts_patterns
        
        # Calculate max atoms across all patterns
        max_atoms = 0
        for s in smarts_patterns:
            mol = Chem.MolFromSmarts(s)
            if mol:
                num_atoms = mol.GetNumAtoms()
                max_atoms = max(max_atoms, num_atoms)
        self._max_atoms = max_atoms

    def forward(self, data: Data) -> Data:
        mol = Chem.MolFromSmiles(data.smiles)
        substructure_instances = []

        if mol is not None:
            for i, smarts in enumerate(self._smarts_patterns):
                pattern = Chem.MolFromSmarts(smarts)
                instances = mol.GetSubstructMatches(pattern)
                instances = [[i] + list(instance) + [-1] * (self._max_atoms - len(instance)) for instance in instances]
                substructure_instances.extend(instances)

        data["substructure_instances"] = torch.tensor(substructure_instances, dtype=torch.long)
        data["n_substructure_instances"] = torch.tensor(len(substructure_instances), dtype=torch.long)
        return data

def get_qm9_smarts_patterns():
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

    return hydrocarbons 
    # return hydrocarbons + haloalkanes + oxygen_containing + nitrogen_containing

