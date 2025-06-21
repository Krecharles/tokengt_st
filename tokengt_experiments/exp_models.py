import torch
import torch.nn as nn
from torch import Tensor, scatter
from typing import Optional
from torch_geometric.nn import TokenGT
from models.token_gt_st_sum import TokenGTST_Sum
from models.token_gt_st_hyp import TokenGTST_Hyp
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import GATConv


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
                                      batch.edge_attr.float(),
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
        batch_norm,
        device, 
    ):
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bns = nn.ModuleList()
            for i in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.bn_final = nn.BatchNorm1d(hidden_channels)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

        # 1-hot encode + linear node features
        self.atom_encoder = nn.Embedding(
            num_embeddings = 28, # num different atoms in ZINC
            embedding_dim = hidden_channels
        )

        self.to(device)
        
        print(f"initialized GCN({num_layers} layers, {hidden_channels} hidden, batch_norm={batch_norm})")

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # 1-hot encode + linear node features
        x = torch.squeeze(self.atom_encoder(x))

        x = self.conv1(x, edge_index)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch_idx)

        if self.batch_norm:
            x = self.bn_final(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


class MPNNConv(MessagePassing):
    """Message Passing Neural Network convolution layer."""
    
    def __init__(self, in_channels, out_channels, edge_channels=None):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels or in_channels
        
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + self.edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), self.edge_channels, device=x.device)
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(inputs)
    
    def update(self, aggr_out, x):
        inputs = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(inputs)


class MPNNGraphRegression(nn.Module):
    """Message Passing Neural Network for graph regression."""

    def __init__(
        self,
        dim_node,
        hidden_channels,
        num_layers,
        dim_edge,
        dropout,
        device,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.conv1 = MPNNConv(dim_node, hidden_channels, dim_edge)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(MPNNConv(hidden_channels, hidden_channels, dim_edge))
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
        self.to(device)
        
        print(f"initialized MPNN({num_layers} layers, {hidden_channels} hidden)")

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = (
            batch.x.float(), 
            batch.edge_index, 
            batch.edge_attr.float() if hasattr(batch, 'edge_attr') and batch.edge_attr is not None else None,
            batch.batch
        )

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch_idx)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x


class GATGraphRegression(nn.Module):
    """Graph Attention Network for graph regression."""

    def __init__(
        self,
        dim_node,
        hidden_channels,
        num_layers,
        heads,
        dropout,
        device,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        self.conv1 = GATConv(dim_node, hidden_channels, heads=heads, dropout=dropout)
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )

        self.lin1 = nn.Linear(hidden_channels * heads, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        
        self.to(device)

        print(
            f"initialized GAT({num_layers} layers, {hidden_channels} hidden, {heads} heads)")

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x.float(), batch.edge_index, batch.batch

        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch_idx)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x

