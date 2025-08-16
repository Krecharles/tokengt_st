import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_geometric.nn.models import GCN
from torch_geometric.nn import global_mean_pool

class GCNGraphRegression(pl.LightningModule):
    def __init__(
        self,
        num_node_features,
        num_substructures,
        hidden_channels=128,
        num_layers=4,
        dropout=0.1,
        lr=0.01,
        batch_size=512,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.atom_encoder = nn.Embedding(num_node_features, hidden_channels)
        self.substructure_encoder = nn.Linear(num_substructures, hidden_channels)

        self.gcn = GCN(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout,
            act="relu",
            norm="batch_norm",
        )
        
        self.lm = nn.Linear(hidden_channels, 1)
        self.criterion = nn.L1Loss()

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        x = self.atom_encoder(x[:,0].long()) + self.substructure_encoder(x[:,1:].float())

        x = self.gcn(x, edge_index)
        
        graph_emb = global_mean_pool(x, batch_idx)

        out = self.lm(graph_emb).squeeze()
        return out

    def _common_step(self, batch):
        out = self(batch)
        loss = self.criterion(out, batch.y)
        return loss

    def training_step(self, batch, batch_idx):
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.log({"learning_rate": lr})
        loss = self._common_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
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
