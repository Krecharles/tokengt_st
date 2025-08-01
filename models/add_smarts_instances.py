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
                    pattern = Chem.MolFromSmarts(smarts)
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
    # fragments = [
    #     # 1. Oxygen-containing carbonyls
    #     [
    #         "*-C(=O)-[C;D1]",       # terminal aldehyde -0.02 (#3)
    #         "*-C(=O)-[N;D1]",       # amide -0.02 (#4)
    #         "*=[O;D1]"              # side-chain aldehydes or ketones -0.42 (#35)
    #     ],

    #     # 2. Nitrogen-based groups
    #     [
    #         "*-[N;D1]",             # primary amines -0.15 (#36)
    #         "*#[N;D1]"              # nitriles -0.13 (#38)
    #     ],

    #     # 3. Unsaturated hydrocarbons
    #     [
    #         "[C]=[C]",              # alkene -0.14 (#39)
    #         "[C]#[C]",              # alkyne -0.15 (#41)
    #         "*-[C;D2]#[C;D1;H]"     # acetylenes -0.11 (#30)
    #     ],

    #     # 4. Alkoxy and hydroxyl groups
    #     [
    #         "*-[O;D2]-[C;D2]-[C;D1;H3]",  # ethoxy -0.01 (#32)
    #         "*-[O;D2]-[C;D1;H3]",         # methoxy -0.07 (#33)
    #         "*-[O;D1]"                    # side-chain hydroxyls -0.39 (#34)
    #     ],

    #     # 5. Miscellaneous groups
    #     [
    #         "*-[#9,#17,#35,#53]",         # halogens -0.02 (#27)
    #     ],
    # ]
    mgssl_motifs = [
        'CCN',
        'NC=O',
        'C1=CC=CC=C1',
        'NC1=CC=CC=C1',
        'CCC',
        'CC1=CC=CC=C1',
        'C1=CC=NC=C1',
        'OC1=CC=CC=C1',
        'CCO',
        'NCCO',
        'COC',
        'O=CO',
        'CN1CCCCC1',
        'C1CCNCC1',
        'CCS',
        'CC[NH3+]',
        'NCN',
        "C=C",
        "C=O",
    ]
    mgssl_motifs = [ [m] for m in mgssl_motifs ]
    return mgssl_motifs


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