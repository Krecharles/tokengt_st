import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv.gcn_conv import GCNConv

from models.token_gt_st_sum import TokenGT, TokenGTST_Sum

class TokenGTGraphRegression(pl.LightningModule):
    def __init__(
        self,
        d_p,
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
        self.atom_encoder = nn.Linear(11, d)
        self.edge_encoder = nn.Linear(4, d)
        self.dist_encoder = nn.Linear(1, d)
        
        self.criterion = nn.L1Loss()

    def forward(self, batch):
        # QM9 node features:
        # [...one_hot(type_idx), atomic_number, aromatic, sp, sp2, sp3, num_hs]
        # all binary except atomic_number and num_hs


        x = self.atom_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)

        pos = batch.pos
        dist = torch.norm(pos[batch.edge_index[0]] - pos[batch.edge_index[1]], dim=1)
        dist_emb = self.dist_encoder(dist.unsqueeze(-1))
        edge_attr = edge_attr + dist_emb

        _, graph_emb = self._token_gt(x,
                                      batch.edge_index,
                                      edge_attr,
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids if "node_ids" in batch else None)
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
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        include_graph_token,
        use_laplacian_node_ids,
        dropout,
        n_substructures,
        lr=0.001,
        target_idx=0,
        batch_size=512,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self._token_gt = TokenGTST_Sum(
            dim_node=d, # because the 1-hot encoding expands the node features to d
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dim_edge=d, # because the 1-hot encoding expands the edge features to d
            use_laplacian_node_ids=use_laplacian_node_ids,
            include_graph_token=include_graph_token,
            dropout=dropout,
            n_substructures=n_substructures,
        )
        self.atom_encoder = nn.Linear(11, d)
        self.edge_encoder = nn.Linear(4, d)

        self.lm = nn.Linear(d, 1)
        
        self.criterion = nn.L1Loss()
        
        print(f"initialized TokenGTST_Sum({n_substructures})")

    def forward(self, batch):
        x = self.atom_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)

        _, graph_emb = self._token_gt(x,
                                      batch.edge_index,
                                      edge_attr,
                                      batch.ptr,
                                      batch.batch,
                                      batch.node_ids if self.hparams["use_laplacian_node_ids"] else None,
                                      batch.substructure_instances,
                                      batch.n_substructure_instances)
        return self.lm(graph_emb).squeeze()

    def training_step(self, batch, batch_idx):
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        return optimizer

class GCNGraphRegression(pl.LightningModule):

    def __init__(
        self,
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

        self.atom_encoder = nn.Linear(11, hidden_channels)

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

    def training_step(self, batch, batch_idx):
        # print(f"training_step {batch.shape}")
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        target = batch.y[:, self.hparams["target_idx"]]
        loss = self.criterion(out, target)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        return optimizer
