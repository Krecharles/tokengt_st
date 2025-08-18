from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch


class AddSubstructureMatchesAsVNs(BaseTransform):
    
    def __init__(self, n_substructures, num_atoms):
        self.n_substructures = n_substructures
        self.num_atoms = num_atoms

    def forward(self, data) -> Data:
        if len(data.substructure_instances) == 0:
            return data

        keys = data.substructure_instances[:, 0]
        vertices = data.substructure_instances[:, 1:]

        num_vertices = data.x.shape[0]
        
        sub_embs = torch.zeros(keys.shape[0], data.x.shape[1])
        sub_embs[:,0] = keys+self.num_atoms
        data.x = torch.cat([data.x, sub_embs], dim=0)
        
        mask = vertices != -1
        vn_indices = torch.arange(keys.shape[0]) + num_vertices
        vn_repeat = vn_indices.unsqueeze(1).expand_as(vertices)
        src = vn_repeat[mask]
        dst = vertices[mask]

        vn_to_v = torch.stack([src, dst], dim=0)
        v_to_vn = torch.stack([dst, src], dim=0)

        data.edge_index = torch.cat([data.edge_index, vn_to_v, v_to_vn], dim=1)

        return data

class AddSubstructureMatchesAsVNsFullyConnected(BaseTransform):
    
    def __init__(self, n_substructures, num_atoms):
        self.n_substructures = n_substructures
        self.num_atoms = num_atoms

    def forward(self, data) -> Data:
        if len(data.substructure_instances) == 0:
            return data

        keys = data.substructure_instances[:, 0]
        vertices = data.substructure_instances[:, 1:]

        num_vertices = data.x.shape[0]
        
        sub_embs = torch.zeros(keys.shape[0], data.x.shape[1])
        sub_embs[:,0] = keys+self.num_atoms
        data.x = torch.cat([data.x, sub_embs], dim=0)
        
        mask = vertices != -1
        vn_indices = torch.arange(keys.shape[0]) + num_vertices
        vn_repeat = vn_indices.unsqueeze(1).expand_as(vertices)
        src = vn_repeat[mask]
        dst = vertices[mask]

        vn_to_v = torch.stack([src, dst], dim=0)
        v_to_vn = torch.stack([dst, src], dim=0)

        if vn_indices.numel() > 1:
            pairs = torch.combinations(vn_indices, r=2)
            fc_1 = torch.stack([pairs[:, 0], pairs[:, 1]], dim=0)
            fc_2 = torch.stack([pairs[:, 1], pairs[:, 0]], dim=0)
            vn_complete = torch.cat([fc_1, fc_2], dim=1)
        else:
            vn_complete = data.edge_index.new_empty((2, 0))        

        data.edge_index = torch.cat([data.edge_index, vn_to_v, v_to_vn, vn_complete], dim=1)

        return data


class AddSubstructureMatchesAsVNsSharingConstituentConnected(BaseTransform):
    
    def __init__(self, n_substructures, num_atoms):
        self.n_substructures = n_substructures
        self.num_atoms = num_atoms

    def forward(self, data) -> Data:
        if len(data.substructure_instances) == 0:
            return data

        keys = data.substructure_instances[:, 0]
        vertices = data.substructure_instances[:, 1:]

        num_vertices = data.x.shape[0]
        
        sub_embs = torch.zeros(keys.shape[0], data.x.shape[1])
        sub_embs[:,0] = keys+self.num_atoms
        data.x = torch.cat([data.x, sub_embs], dim=0)
        
        mask = vertices != -1
        vn_indices = torch.arange(keys.shape[0]) + num_vertices
        vn_repeat = vn_indices.unsqueeze(1).expand_as(vertices)
        src = vn_repeat[mask]
        dst = vertices[mask]

        vn_to_v = torch.stack([src, dst], dim=0)
        v_to_vn = torch.stack([dst, src], dim=0)

        shared_pairs_undirected = []
        if src.numel() > 0:
            # group the vn the atoms they touch
            uniq_atoms, inv = torch.unique(dst, return_inverse=True)
            for g in range(uniq_atoms.numel()):
                vn_grp = src[inv == g]         
                vn_grp = torch.unique(vn_grp)   # de-dup VNs if same VN matched atom multiple times
                if vn_grp.numel() >= 2:
                    pairs = torch.combinations(vn_grp, r=2)    
                    # canonical ordering (min, max) so can be unique across atoms
                    a = torch.minimum(pairs[:, 0], pairs[:, 1])
                    b = torch.maximum(pairs[:, 0], pairs[:, 1])
                    shared_pairs_undirected.append(torch.stack([a, b], dim=0)) 
        
        if shared_pairs_undirected:
            undirected = torch.cat(shared_pairs_undirected, dim=1)
            undirected = torch.unique(undirected.t(), dim=0).t()            
            vn_shared = torch.cat([undirected, undirected.flip(0)], dim=1)     
        else:
            vn_shared = data.edge_index.new_empty((2, 0))

        data.edge_index = torch.cat([data.edge_index, vn_to_v, v_to_vn, vn_shared], dim=1)

        return data

class AddGlobalVN(BaseTransform):
    
    def __init__(self, num_atoms):
        self.num_atoms = num_atoms
    
    def forward(self, data) -> Data:
        
        n, f = data.x.shape[0], data.x.shape[1]
        src = torch.arange(n)

        vn_col = torch.full((n,), n)

        vn_to_v = torch.stack([src, vn_col], dim=0)
        v_to_vn = torch.stack([vn_col, src], dim=0)

        data.edge_index = torch.cat([data.edge_index, vn_to_v, v_to_vn], dim=1)

        sub_embs = torch.zeros(f)
        sub_embs[0] = self.num_atoms
        data.x = torch.cat([data.x, sub_embs.unsqueeze(0)], dim=0)

        return data
