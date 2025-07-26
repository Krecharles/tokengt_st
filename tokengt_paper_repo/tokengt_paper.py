import torch
import torch.nn as nn
import pytorch_lightning as pl

from tokengt_paper_repo import TokenGTGraphEncoder

class TokenGTPaperGraphRegression(pl.LightningModule):
    def __init__(
        self,
        num_atoms,
        num_edges,
        d_p,
        d,
        num_heads,
        num_encoder_layers,
        node_id_mode,
        dropout,
        lr=0.001,
        batch_size=512,
        weight_decay=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._token_gt = TokenGTGraphEncoder(
            # <
            num_atoms=num_atoms,
            num_edges=num_edges,
            # >
            # < for tokenization
            rand_node_id=node_id_mode == "rand",
            rand_node_id_dim=d_p,
            orf_node_id=node_id_mode == "orf",
            orf_node_id_dim=d_p,
            lap_node_id=node_id_mode == "laplacian",
            lap_node_id_k=d_p,
            type_id=True,
            # >
            # <
            stochastic_depth=False,
            performer=False,
            performer_finetune=False,
            performer_nb_features=None,
            performer_feature_redraw_interval=1000,
            performer_generalized_attention=False,

            num_encoder_layers=num_encoder_layers,
            embedding_dim=d,
            ffn_embedding_dim=d,
            num_attention_heads=num_heads,
            dropout=dropout,
            attention_dropout=dropout,
            activation_dropout=dropout,
            encoder_normalize_before=True,
            layernorm_style="postnorm",
            apply_graphormer_init=True,
            activation_fn="gelu",
            return_attention=False,
            # >
        )
        self.lm = nn.Linear(d, 1)
        self.criterion = nn.L1Loss()

    def forward(self, batch):
        _, graph_emb, _ = self._token_gt(batch)
        return self.lm(graph_emb).squeeze()

    def _common_step(self, batch):
        out = self(batch)
        loss = self.criterion(out, batch.y)
        return loss

    def training_step(self, batch, batch_idx):
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        return optimizer
