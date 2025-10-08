"""
Modified from https://github.com/microsoft/Graphormer
"""

import torch
import numpy as np
import torch.nn.functional as F

from . import algos


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


class AddTokenGTPaperNodeIdentifiers:
    def __init__(self, d_p, convert_to_single_emb_offset: int = 512):
        self.d_p = d_p
        self.convert_to_single_emb_offset = convert_to_single_emb_offset

    def __call__(self, item):
        edge_int_feature, edge_index, node_int_feature = item.edge_attr, item.edge_index, item.x
        node_data = convert_to_single_emb(node_int_feature, self.convert_to_single_emb_offset)
        if len(edge_int_feature.size()) == 1:
            edge_int_feature = edge_int_feature[:, None]
        edge_data = convert_to_single_emb(edge_int_feature, self.convert_to_single_emb_offset)

        N = node_int_feature.size(0)
        dense_adj = torch.zeros([N, N], dtype=torch.bool)
        dense_adj[edge_index[0, :], edge_index[1, :]] = True
        in_degree = dense_adj.long().sum(dim=1).view(-1)
        lap_eigvec, lap_eigval = algos.lap_eig(dense_adj, N, in_degree)  # [N, N], [N,]
        lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)

        lap_dim = lap_eigvec.size(-1)
        if self.d_p > lap_dim:
            lap_eigvec = F.pad(lap_eigvec, (0, self.d_p - lap_dim), value=float('0'))  # [sum(n_node), Dl]
        else:
            lap_eigvec = lap_eigvec[:, :self.d_p]  # [sum(n_node), Dl]

        item.node_data = node_data
        item.edge_data = edge_data
        item.edge_index = edge_index
        item.in_degree = in_degree
        item.out_degree = in_degree  # for undirected graph
        item.lap_eigvec = lap_eigvec
        return item
