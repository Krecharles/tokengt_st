from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch

class AddSubstructureEmbeddings(BaseTransform):
    """
    Augments the node features with by n_substructures where the i-th additional 
    feature is the number of time the given node is a member of the i-th substructure.
    """

    def __init__(self, n_substructures, accumulate: bool = False):
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