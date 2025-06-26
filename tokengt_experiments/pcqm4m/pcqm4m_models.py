import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv

from models.token_gt_st_sum import TokenGT

def convert_to_single_emb(x, offset: int = 512):
    # https://github.com/jw9730/tokengt/blob/main/large-scale-regression/tokengt/data/wrapper.py
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

class TokenGTGraphRegression(nn.Module):
    def __init__(
        self,
        d_p,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        include_graph_token,
        is_laplacian_node_ids,
        dropout,
        device,
    ):
        super().__init__()
        self._token_gt = TokenGT(
            dim_node=d,
            dim_edge=d,
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

        # tokengt paper repo uses 512 * 9 as the embedding size.
        self.atom_encoder = nn.Embedding(512 * 9, d, padding_idx=0)
        self.edge_encoder = nn.Embedding(512 * 3, d, padding_idx=0)

    def forward(self, batch):

        x = convert_to_single_emb(batch.x)
        edge_attr = convert_to_single_emb(batch.edge_attr)

        x = self.atom_encoder(x).sum(dim=1) 
        edge_attr = self.edge_encoder(edge_attr).sum(dim=1)

        _, graph_emb = self._token_gt(x,
                                      batch.edge_index,
                                      edge_attr,
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids)
        return self.lm(graph_emb)

class GCNGraphRegression(nn.Module):
    """Graph Convolutional Network for graph regression."""

    def __init__(
        self,
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

        # tokengt paper repo uses 512 * 9 as the embedding size.
        self.atom_encoder = nn.Embedding(512 * 9, hidden_channels, padding_idx=0)
        # GCN does not use edge features, so we don't need to encode them.

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


        self.to(device)
        
        print(f"initialized GCN({num_layers} layers, {hidden_channels} hidden, batch_norm={batch_norm})")

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        x = convert_to_single_emb(x)

        x = self.atom_encoder(x).sum(dim=1)

        x = self.conv1(x.float(), edge_index)
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

class GATGraphRegression(nn.Module):
    """Graph Attention Network for graph regression."""

    def __init__(
        self,
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

        # tokengt paper repo uses 512 * 9 as the embedding size.
        self.atom_encoder = nn.Embedding(512 * 9, hidden_channels, padding_idx=0)
        # GAT does not use edge features, so we don't need to encode them.

        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
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
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        x = convert_to_single_emb(x)
        x = self.atom_encoder(x).sum(dim=1)

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
