import torch
import torch.nn as nn
import pytorch_lightning as pl

from tokengt_pyg import TokenGT


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
        warmup_fraction=0.1,
        min_lr_ratio=0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._token_gt = TokenGT(
            # <
            num_atoms=num_atoms,
            num_edges=num_edges,
            # >
            # < for tokenization
            node_id_mode=node_id_mode,
            d_p=d_p,
            lap_node_id_eig_dropout=0.2,
            # >
            # <
            num_encoder_layers=num_encoder_layers,
            embedding_dim=d,
            ffn_embedding_dim=d,
            num_attention_heads=num_heads,
            dropout=dropout,
            attention_dropout=dropout,
            activation_dropout=dropout,
            norm_first=True,
            activation_fn="gelu",
            # >
        )
        self.lm = nn.Linear(d, 1)
        self.criterion = nn.L1Loss()

    def forward(self, batch):
        node_embs, edge_embs, graph_emb = self._token_gt(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.ptr,
            batch.batch,
            batch.node_ids,
        )
        return self.lm(graph_emb).squeeze()

    def _common_step(self, batch):
        out = self(batch)
        loss = self.criterion(out, batch.y)
        return loss

    def training_step(self, batch, batch_idx):
        lr = self.optimizers().param_groups[0]["lr"]
        self.logger.experiment.log({"learning_rate": lr})

        loss = self._common_step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams["batch_size"],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams["batch_size"],
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams["batch_size"],
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(self.hparams["warmup_fraction"] * total_steps))
        decay_steps = max(1, total_steps - warmup_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.00001,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=self.hparams["min_lr_ratio"] * self.hparams["lr"],
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_then_cosine",
            },
        }
