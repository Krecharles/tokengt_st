"""
Modified from https://github.com/microsoft/Graphormer
"""

from typing import Literal

import torch.nn as nn

from .tokenizer import GraphFeatureTokenizer

class TokenGTGraphEncoder(nn.Module):
    def __init__(
            self,
            num_atoms: int,
            num_edges: int,

            node_id_mode: Literal["orf", "laplacian"],
            d_p: int,

            lap_node_id_sign_flip: bool = False,
            lap_node_id_eig_dropout: float = 0.0,

            num_encoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            encoder_normalize_before: bool = True,
            norm_first: bool = False,
            activation_fn: str = "gelu",

            return_attention: bool = False,
            **transformer_kwargs,

    ) -> None:

        super().__init__()
        # TODO document this dropout as opposed to attention and activation dropouts
        self.dropout_module = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

        self.graph_feature = GraphFeatureTokenizer(
            num_atoms=num_atoms,
            num_edges=num_edges,
            node_id_mode=node_id_mode,
            d_p=d_p,
            hidden_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=ffn_embedding_dim,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
        )

        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        enc_layer = nn.TransformerEncoderLayer(
            embedding_dim,  
            num_attention_heads,
            ffn_embedding_dim,
            dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=activation_fn,
            **transformer_kwargs,
        )
        self._transformer_encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        self.apply(self._init_params)

    def forward(self, batched_data):

        # x: B x T x C
        x, padding_mask, padded_index = self.graph_feature(batched_data)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)
        x = self._transformer_encoder(x, src_key_padding_mask=padding_mask)

        graph_rep = x[:, 0, :]
        return x, graph_rep

    @staticmethod
    def _init_params(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()