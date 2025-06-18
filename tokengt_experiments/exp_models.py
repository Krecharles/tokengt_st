import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from torch_geometric.nn import TokenGT
from models.token_gt_st_sum import TokenGTST_Sum
from models.token_gt_st_hyp import TokenGTST_Hyp
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class TokenGTGraphRegression(nn.Module):
    def __init__(
        self,
        dim_node,
        d_p,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        include_graph_token,
        is_laplacian_node_ids,
        dim_edge,
        dropout,
        device,
    ):
        super().__init__()
        self._token_gt = TokenGT(
            dim_node=dim_node,
            dim_edge=dim_edge,
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            is_laplacian_node_ids=is_laplacian_node_ids,
            include_graph_token=include_graph_token,
            dropout=dropout,
            device=device,
        )
        self.lm = nn.Linear(d, 1, device=device)

    def forward(self, batch):
        _, graph_emb = self._token_gt(batch.x.float(),
                                      batch.edge_index,
                                      batch.edge_attr.unsqueeze(1).float(),
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids)
        return self.lm(graph_emb)


class TokenGTSTSumGraphRegression(nn.Module):
    """TokenGT with substructure tokens using sum aggregation."""

    def __init__(
        self,
        dim_node,
        d_p,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        include_graph_token,
        is_laplacian_node_ids,
        dim_edge,
        dropout,
        device,
        n_substructures
    ):
        super().__init__()
        self._token_gt = TokenGTST_Sum(
            dim_node=dim_node,
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dim_edge=dim_edge,
            is_laplacian_node_ids=is_laplacian_node_ids,
            include_graph_token=include_graph_token,
            dropout=dropout,
            device=device,
            n_substructures=n_substructures
        )
        self.lm = nn.Linear(d, 1, device=device)
        print(f"initialized TokenGTST_Sum({n_substructures})")

    def forward(self, batch, substructure_instances=None):
        _, graph_emb = self._token_gt(batch.x.float(),
                                      batch.edge_index,
                                      batch.edge_attr.unsqueeze(1).float(),
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids,
                                      substructure_instances)
        return self.lm(graph_emb)


class TokenGTSTHypGraphRegression(nn.Module):
    """TokenGT with substructure tokens using hypergraph aggregation."""

    def __init__(
        self,
        dim_node,
        d_p,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        include_graph_token,
        is_laplacian_node_ids,
        dim_edge,
        dropout,
        device,
        n_substructures
    ):
        super().__init__()
        self._token_gt = TokenGTST_Hyp(
            dim_node=dim_node,
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dim_edge=dim_edge,
            is_laplacian_node_ids=is_laplacian_node_ids,
            include_graph_token=include_graph_token,
            dropout=dropout,
            device=device,
            n_substructures=n_substructures
        )
        self.lm = nn.Linear(d, 1, device=device)
        print(f"initialized TokenGTST_Hyp({n_substructures})")

    def forward(self, batch, substructure_instances=None):
        _, graph_emb = self._token_gt(batch.x.float(),
                                      batch.edge_index,
                                      batch.edge_attr.unsqueeze(1).float(),
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids,
                                      substructure_instances)
        return self.lm(graph_emb)


class GCNGraphRegression(nn.Module):
    """Graph Convolutional Network for graph regression."""

    def __init__(
        self,
        dim_node,
        hidden_channels,
        num_layers,
        dropout,
        device,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.conv1 = GCNConv(dim_node, hidden_channels)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
        # Move model to device
        self.to(device)
        
        print(
            f"initialized GCN({num_layers} layers, {hidden_channels} hidden)")

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x.float(), batch.edge_index, batch.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch_idx)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x
