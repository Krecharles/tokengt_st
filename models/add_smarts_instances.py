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

def get_pcqm4m_smarts_patterns():
    motifs = [
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
    motifs = [ [m] for m in motifs ]
    return motifs

def get_pcqm4m_xl_smarts_patterns():
    mgssl_motifs_xl = ['CC',
        'CN',
        'N=O',
        'CCN',
        '[NH3+]C1=CC=CC=C1',
        'NO',
        'CNO',
        '[NH3+][O-]',
        'C1=CC=CC=C1',
        '[NH2+]=O',
        'NC1=CC=CC=C1',
        '[NH3+]O',
        'N[O-]',
        'C[NH3+]',
        'CC1=CC=CC=C1',
        'CC1CCC1',
        'C=O',
        'CCC=O',
        'CCCCC',
        'CCCCCC',
        'C1CCC1',
        'CCCC',
        'CC=O',
        'CCC',
        'OC1=CC=CC=C1',
        'CO',
        'CCO',
        'COC',
        'CCCO',
        'CF',
        'C=N',
        'CC=N',
        'C=NC',
        'N=CO',
        'CS',
        'CCS',
        'CCCS',
        'O=CO',
        'C1=C[NH]C=C1',
        'CC1=CC=C[NH]1',
        'CC#N',
        'C#N',
        'CON',
        'C1=C[NH]C=N1',
        'C1=CN=CN=C1',
        'CCCCO',
        'CCCCCO',
        'C#CC',
        'C#CCC',
        'CCCCCCCC',
        'C#C',
        'CCNO',
        'CC=NO',
        'C=NO',
        'NCCO',
        'NCN',
        'CCCN',
        'C=CCCC',
        'CC1CCCO1',
        'C=CCCCCC',
        'C=CCC',
        'C1CCOC1',
        'C=C',
        'C1=CCCCC1',
        'C=CC',
        'NC=O',
        'CCCCS',
        'OC1CCCCC1',
        'CN1CCCC1',
        'CC1CCCN1',
        'C1CCNC1',
        'C1CCCCC1',
        'CC1CCCCC1',
        'NC1CCCCC1',
        'CCC#N',
        'CCl',
        'NC1=NC=CC=C1',
        'C1=CC=NC=C1',
        'C1CCNCC1',
        'NC1=CC=CC=N1',
        'C1=C[NH]N=C1',
        'C=CCN',
        'OCO',
        'CBr',
        'C1CNCCN1',
        'CN1CCNCC1',
        'CN1CCCCC1',
        'CC1CCCNC1',
        'C1=CSC=N1',
        'CC1=CSC=C1',
        'NC1CC1',
        'C=NCC',
        'C1CC1',
        'C1=CSC=C1',
        'COCO',
        'CCCF',
        'CCF',
        'CNN',
        'NN',
        'C1=NCNN1',
        'CN1C=CC=N1',
        'CC1=N[NH]C=C1',
        'CC1=CC=N[NH]1',
        'CC1=C[NH]N=C1',
        'CC1CC1',
        'CCCC=N',
        'C=NCCC',
        'C1CCOCC1',
        'NCO',
        'CCCl',
        'C=CO',
        'CC1=CC=CO1',
        'C1=COC=C1',
        'NCS',
        'CC1=NC=CC=C1',
        'CC1=CC=CC=N1',
        'C=CN',
        'CC1=CC=CS1',
        'CN1C=CN=C1',
        'CC1=CN=C[NH]1',
        'CC1=C[NH]C=N1',
        'CC1=NC=NC=C1',
        'CC1=CC=NC=N1',
        'CC1CNCCN1',
        'CC1=CC=CN=C1',
        'CC1CCCC1',
        'C1CCCC1',
        'CC1CCCC(C)C1',
        'CC=CN',
        'C=NN',
        'CC1CCNC1',
        'C1CNC1',
        'CCCC#N',
        'NS',
        'CC#CC',
        'C1=CN=CC=N1',
        'C1CCCCCC1',
        'CC1=CC=NC=C1',
        'NC1CCCC1',
        'CCCCl',
        'CC1CCNCC1',
        'CC1=NC=C[NH]1']
    mgssl_motifs_xl = [ [m] for m in mgssl_motifs_xl ]
    return mgssl_motifs_xl



# class ConvertToSingleEmb(BaseTransform):
#     # offsets the node features by 1 so that a node feature of 0 is meaningless (padding_idx=0 in the embedding)
#     # takes the max_value 

#     def __init__(self, offset: int = 512):
#         self.offset = offset

#     def forward(self, data) -> Data:
#         data.x = convert_to_single_emb(data.x, self.offset)
#         data.edge_data = convert_to_single_emb(data.edge_data, self.offset)
#         return data