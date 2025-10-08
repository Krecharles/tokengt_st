from rdkit import Chem
from rdkit.Chem import Draw
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        substructure_mode=None,
        n_substructures=0,
        return_attention=False,
        use_interaction_bias=False,
        warmup_fraction=0.1,
        min_lr_ratio=0.05,
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
            substructure_mode=substructure_mode,
            n_substructures=n_substructures,
            return_attention=return_attention,
            use_interaction_bias=use_interaction_bias,
            # >
        )
        self.lm = nn.Linear(d, 1)
        self.criterion = nn.L1Loss()

    def forward(self, batch):
        _, graph_emb, attn_dict = self._token_gt(batch)
        return self.lm(graph_emb).squeeze(), attn_dict

    def _common_step(self, batch, log_attention=False):
        out, attn_dict = self(batch)
        loss = self.criterion(out, batch.y)
        if log_attention and self.hparams["return_attention"]:
            self.log_sample(attn_dict, batch, batch.y, out)
        return loss

    def log_sample(self, attn_dict, batch, y_true, y_pred):
        # logs the first graph in the batch with attention maps in every layer and head
        cols = ["smiles", "mol", "y_true", "y_pred", "n_atoms", "n_edges", "n_substructure_instances", "node_features", "edge_features", "substructure_instances"]
        for layer in range(len(attn_dict["maps"])):
            for head in range(len(attn_dict["maps"][layer])):
                cols.append(f"attn_maps_{layer}_{head}")
        table = wandb.Table(columns=cols)
        
        prev_n_nodes = 0
        prev_n_edges = 0
        prev_n_substructures = 0

        for i in tqdm(range(10), desc="Logging samples"):
            mol = Chem.MolFromSmiles(batch.smiles[i])
            img = Draw.MolToImage(mol, size=(300, 300))
            row = [batch.smiles[i], wandb.Image(img), y_true[i], y_pred[i]]
            
            n_nodes = batch.ptr[i+1] - batch.ptr[i]
            edge_num = torch.bincount(batch.batch[batch.edge_index[0]], minlength=int(batch.batch.max()) + 1)
            n_edges = edge_num[i]
            n_substructures = batch.n_substructure_instances[i]

            row.append(n_nodes)
            row.append(n_edges)
            row.append(n_substructures)
            row.append(str(batch.node_data[prev_n_nodes:prev_n_nodes+n_nodes].detach().cpu().numpy().tolist()))
            row.append(str(batch.edge_data[prev_n_edges:prev_n_edges+n_edges].detach().cpu().numpy().tolist()))
            row.append(str(batch.substructure_instances[prev_n_substructures:prev_n_substructures+n_substructures].detach().cpu().numpy().tolist()))

            prev_n_nodes += n_nodes
            prev_n_edges += n_edges
            prev_n_substructures += n_substructures

            for layer in range(len(attn_dict["maps"])):
                for head in range(len(attn_dict["maps"][layer])):
                    fig = self.create_attention_heatmap(attn_dict["maps"][layer][head][i], layer, head) # 0th graph in batch
                    row.append(wandb.Image(fig))
                    plt.close(fig)

            table.add_data(*row)
        wandb.log({"sample": table}, step=self.global_step+1)

    def create_attention_heatmap(self, attention, layer, head):
        fig, ax = plt.subplots(figsize=(4, 4))
        cax = ax.matshow(attention.detach().cpu().numpy(), cmap="viridis")
        ax.set_title(f"Layer {layer}, Head {head}")
        plt.tight_layout()
        return fig

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
        loss = self._common_step(batch, log_attention=batch_idx == 0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])

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
