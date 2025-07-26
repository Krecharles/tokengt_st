import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.message_passing import MessagePassing

from models.token_gt_st_sum import TokenGT, TokenGTST_Sum

class TokenGTGraphRegression(pl.LightningModule):
    def __init__(
        self,
        d_p,
        dim_node,
        dim_edge,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        include_graph_token,
        node_id_mode,
        dropout,
        lr=0.001,
        target_idx=0,
        batch_size=512,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._token_gt = TokenGT(
            dim_node=d,
            dim_edge=d,
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            include_graph_token=include_graph_token,
            node_id_mode=node_id_mode,
            dropout=dropout,
        )
        self.lm = nn.Linear(d, 1)

        # Since most features are binary, no need for embedding.
        self.atom_encoder = nn.Linear(dim_node, d)
        self.edge_encoder = nn.Linear(dim_edge, d)
        self.dist_encoder = nn.Linear(1, d)
        
        self.criterion = nn.L1Loss()

    def forward(self, batch):
        # QM9 node features:
        # [...one_hot(type_idx), atomic_number, aromatic, sp, sp2, sp3, num_hs]
        # all binary except atomic_number and num_hs


        x = self.atom_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)

        dist = torch.norm(batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]], dim=1)
        dist_emb = self.dist_encoder(dist.unsqueeze(-1))
        edge_attr = edge_attr + dist_emb

        _, graph_emb = self._token_gt(x,
                                      batch.edge_index,
                                      edge_attr,
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids if hasattr(batch, "node_ids") else None)
        return self.lm(graph_emb).squeeze()

    def _common_step(self, batch):
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        return optimizer

class TokenGTSTSumGraphRegression(pl.LightningModule):

    def __init__(
        self,
        d_p,
        dim_node,
        dim_edge,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        include_graph_token,
        node_id_mode,
        dropout,
        n_substructures,
        lr=0.001,
        target_idx=0,
        batch_size=512,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self._token_gt = TokenGTST_Sum(
            d_p=d_p,
            dim_node=d,
            dim_edge=d,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            node_id_mode=node_id_mode,
            include_graph_token=include_graph_token,
            dropout=dropout,
            n_substructures=n_substructures,
        )
        self.atom_encoder = nn.Linear(dim_node, d)
        self.edge_encoder = nn.Linear(dim_edge, d)
        self.dist_encoder = nn.Linear(1, d)

        self.lm = nn.Linear(d, 1)
        
        self.criterion = nn.L1Loss()
        
        print(f"initialized TokenGTST_Sum({n_substructures})")

    def forward(self, batch):
        x = self.atom_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)

        dist = torch.norm(batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]], dim=1)
        dist_emb = self.dist_encoder(dist.unsqueeze(-1))
        edge_attr = edge_attr + dist_emb

        _, graph_emb = self._token_gt(x,
                                      batch.edge_index,
                                      edge_attr,
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids if hasattr(batch, "node_ids") else None,
                                      batch.substructure_instances,
                                      batch.n_substructure_instances)
        return self.lm(graph_emb).squeeze()

    def _common_step(self, batch):
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        return optimizer

class GCNGraphRegression(pl.LightningModule):

    def __init__(
        self,
        dim_node,
        hidden_channels,
        num_layers,
        dropout,
        batch_norm,
        lr=0.001,
        target_idx=0,
        batch_size=512,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.atom_encoder = nn.Linear(dim_node, hidden_channels)

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

        self.criterion = nn.L1Loss()

        print(f"initialized GCN({num_layers} layers, {hidden_channels} hidden, batch_norm={batch_norm})")

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        x = self.atom_encoder(x)

        x = self.conv1(x.float(), edge_index)
        if self.hparams["batch_norm"]:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.hparams["dropout"], training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.hparams["batch_norm"]:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.hparams["dropout"], training=self.training)

        x = global_mean_pool(x, batch_idx)

        if self.hparams["batch_norm"]:
            x = self.bn_final(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.hparams["dropout"], training=self.training)
        x = self.lin2(x)

        return x.squeeze()

    def _common_step(self, batch):
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        return optimizer


class MPNNGraphRegression(pl.LightningModule):

    def __init__(
        self,
        dim_node,
        dim_edge,
        hidden_channels,
        num_layers,
        dropout,
        batch_norm,
        lr=0.001,
        target_idx=0,
        batch_size=512,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.atom_encoder = nn.Linear(dim_node, hidden_channels)
        self.edge_encoder = nn.Linear(dim_edge, hidden_channels)
        self.dist_encoder = nn.Linear(1, hidden_channels)

        # MPNN layers
        self.conv1 = MPNNConv(hidden_channels, hidden_channels, edge_dim=hidden_channels)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(MPNNConv(hidden_channels, hidden_channels, edge_dim=hidden_channels))
        
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bns = nn.ModuleList()
            for i in range(num_layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.bn_final = nn.BatchNorm1d(hidden_channels)
        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

        self.criterion = nn.L1Loss()

        print(f"initialized MPNN({num_layers} layers, {hidden_channels} hidden, batch_norm={batch_norm})")

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        x = self.atom_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        dist = torch.norm(batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]], dim=1)
        dist_emb = self.dist_encoder(dist.unsqueeze(-1))
        edge_attr = edge_attr + dist_emb

        x = self.conv1(x, edge_index, edge_attr)
        if self.hparams["batch_norm"]:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.hparams["dropout"], training=self.training)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if self.hparams["batch_norm"]:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.hparams["dropout"], training=self.training)

        x = global_mean_pool(x, batch_idx)

        if self.hparams["batch_norm"]:
            x = self.bn_final(x)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.hparams["dropout"], training=self.training)
        x = self.lin2(x)

        return x.squeeze()

    def _common_step(self, batch):
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        return optimizer


class MPNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=0):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is not None:
            message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            message_input = torch.cat([x_i, x_j], dim=-1)
        return self.message_mlp(message_input)

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)

