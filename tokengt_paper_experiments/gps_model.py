# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_gps.py

import os.path as osp
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention


class GPS(pl.LightningModule):
    def __init__(self, 
                 channels: int = 64, 
                 pe_channels: int = 20,
                 pe_dim: int = 8, 
                 num_layers: int = 10,
                 attn_type: str = 'multihead',
                 attn_kwargs: Dict[str, Any] = None,
                 lr: float = 0.001,
                 weight_decay: float = 1e-5,
                 batch_size: int = 32,
                 num_atoms: int = 30,
                 num_edges: int = 4):
        super().__init__()
        self.save_hyperparameters()

        self.atom_encoder = nn.Embedding(num_atoms, channels - pe_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(num_edges, channels, padding_idx=0)

        self.pe_lin = Linear(pe_channels, pe_dim)
        self.pe_norm = BatchNorm1d(pe_channels)
        # self.node_emb = Embedding(28, channels - pe_dim)
        # self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn_seq = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn_seq), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
        
        self.criterion = nn.L1Loss()

    def forward(self, batch):

        node_data = batch["node_data"]
        pe = batch["pe"]
        edge_index = batch["edge_index"]
        edge_data = batch["edge_data"]
        batch = batch["batch"]

        x_pe = self.pe_norm(pe)

        node_feature = self.atom_encoder(node_data.int()).sum(-2)
        edge_feature = self.edge_encoder(edge_data.int()).sum(-2)

        x = torch.cat((node_feature, self.pe_lin(x_pe)), 1)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_feature)
        x = global_add_pool(x, batch)
        return self.mlp(x)

    def _common_step(self, batch, batch_idx):
        data = batch
        self.redraw_projection.redraw_projections()
        out = self(data)
        loss = self.criterion(out.squeeze(), data.y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, 
                batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=self.hparams.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, 
                batch_size=self.hparams.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        return optimizer


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
