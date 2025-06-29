import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import TokenGT
import torch.nn.functional as F
from torch_geometric.nn import GPSConv
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_add_pool

from models.token_gt_st_sum import TokenGTST_Sum
from models.token_gt_st_hyp import TokenGTST_Hyp


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
        use_one_hot_encoding,
        num_embeddings,
        dim_edge,
        dropout,
        device,
    ):
        super().__init__()
        self.use_one_hot_encoding = use_one_hot_encoding

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
        if self.use_one_hot_encoding:
            # 1-hot encode
            self.atom_encoder = nn.Embedding(
                num_embeddings = num_embeddings, # num different atoms in ZINC
                embedding_dim = num_embeddings
            )
        self.lm = nn.Linear(d, 1, device=device)
        print(f"initialized TokenGT({dim_node} node features, {dim_edge} edge features, {d} hidden, {num_heads} heads, {num_encoder_layers} layers, {dim_feedforward} feedforward, {include_graph_token} graph token, {is_laplacian_node_ids} laplacian node ids, {use_one_hot_encoding} one hot encoding, {dropout} dropout)")

    def forward(self, batch):
        if self.use_one_hot_encoding:
            # atom features are the first column of x, other features come later
            atom_features = torch.squeeze(self.atom_encoder(batch.x[:, 0].long()))
            batch.x = torch.cat([atom_features, batch.x[:, 1:]], dim=1)

        _, graph_emb = self._token_gt(batch.x.float(),
                                      batch.edge_index,
                                      None,
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
        use_one_hot_encoding,
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
        self.use_one_hot_encoding = use_one_hot_encoding

        if use_one_hot_encoding:
            # 1-hot encode
            self.atom_encoder = nn.Embedding(
                num_embeddings = 28,
                embedding_dim = 28
            )

        self.lm = nn.Linear(d, 1, device=device)
        print(f"initialized TokenGTST_Sum({n_substructures})")

    def forward(self, batch):
        if self.use_one_hot_encoding:
            # atom features are the first column of x, other features come later
            atom_features = torch.squeeze(self.atom_encoder(batch.x[:, 0].long()))
        else:
            # throw away count embedding in tgt_sum
            atom_features = batch.x[:, 0].unsqueeze(1).float()
        _, graph_emb = self._token_gt(atom_features,
                                      batch.edge_index,
                                      None,
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids,
                                      batch.substructure_instances,
                                      batch.n_substructure_instances)
        return self.lm(graph_emb)

class TokenGTSTHypGraphRegression(nn.Module):
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
        use_one_hot_encoding,
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
        self.use_one_hot_encoding = use_one_hot_encoding

        if use_one_hot_encoding:
            # 1-hot encode
            self.atom_encoder = nn.Embedding(
                num_embeddings = 28,
                embedding_dim = 28
            )

        self.lm = nn.Linear(d, 1, device=device)
        print(f"initialized TokenGTST_Sum({n_substructures})")

    def forward(self, batch):
        if self.use_one_hot_encoding:
            # atom features are the first column of x, other features come later
            atom_features = torch.squeeze(self.atom_encoder(batch.x[:, 0].long()))
        else:
            # throw away count embedding in tgt_sum
            atom_features = batch.x[:, 0].unsqueeze(1).float()
        _, graph_emb = self._token_gt(atom_features,
                                      batch.edge_index,
                                      None,
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids,
                                      batch.substructure_instances,
                                      batch.n_substructure_instances)
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
        use_one_hot_encoding,
        device, 
        num_embeddings
    ):
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.use_one_hot_encoding = use_one_hot_encoding

        if self.use_one_hot_encoding:
            self.atom_encoder = nn.Embedding(
                num_embeddings = num_embeddings,
                embedding_dim = hidden_channels,
            )
            self.mlp = nn.Linear(dim_node-1, hidden_channels)
            self.conv1 = GCNConv(hidden_channels, hidden_channels)
        else:
            self.conv1 = GCNConv(dim_node, hidden_channels)

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
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # 1-hot encode + linear node features
        if self.use_one_hot_encoding:
            # atom features are the first column of x, other features come later
            atom_features = torch.squeeze(self.atom_encoder(x[:, 0].long()))
            count_features = self.mlp(x[:, 1:])
            x = atom_features + count_features

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

class GPS(torch.nn.Module):
    # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/graph_transformer.html
    # def __init__(self, channels: int, pe_dim: int, num_layers: int,
    #             attn_type: str, attn_kwargs: dict, batch_norm: bool = True,
    #             use_one_hot_encoding: bool = False, dim_node: int = 28, num_embeddings: int = 28):
    def __init__(self, 
                dim_node: int,
                hidden_channels: int,
                num_layers: int,
                batch_norm: bool = True,
                use_one_hot_encoding: bool = False,
                num_embeddings: int = 28,
                device: str = "cuda"):
        super().__init__()

        self.batch_norm = batch_norm
        self.use_one_hot_encoding = use_one_hot_encoding

        if self.use_one_hot_encoding:
            self.atom_encoder = nn.Embedding(
                num_embeddings = num_embeddings,
                embedding_dim = hidden_channels,
            )
            self.mlp = nn.Linear(dim_node-1, hidden_channels)
        else:
            self.mlp = nn.Linear(dim_node, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            conv = GPSConv(hidden_channels, GINEConv(mlp), heads=4,
                        attn_type="multihead")
            self.convs.append(conv)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

        if batch_norm:
            self.bn_final = nn.BatchNorm1d(hidden_channels)

        self.to(device)

        print(f"initialized GPS({num_layers} layers, {hidden_channels} hidden, batch_norm={batch_norm})")

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        if self.use_one_hot_encoding:
            # atom features are the first column of x, other features come later
            atom_features = torch.squeeze(self.atom_encoder(x[:, 0].long()))
            count_features = self.mlp(x[:, 1:].float())
            x = atom_features + count_features
        else:
            x = self.mlp(x.float())

        # add zero edge features
        edge_attr = torch.zeros(edge_index.shape[1], x.shape[1], device=x.device)
        for conv in self.convs:
            x = conv(x, edge_index, batch_idx, edge_attr=edge_attr)
        
        x = global_mean_pool(x, batch_idx)
        
        if self.batch_norm:
            x = self.bn_final(x)
        
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        return x