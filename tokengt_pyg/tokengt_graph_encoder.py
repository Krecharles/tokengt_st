"""
Modified from https://github.com/microsoft/Graphormer
"""

from typing import Literal, Optional

import torch
import torch.nn as nn

from .multihead_attention import MultiheadAttention
from .tokenizer import GraphFeatureTokenizer
from .tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


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
            apply_graphormer_init: bool = False,
            activation_fn: str = "gelu",

            return_attention: bool = False,
            **transformer_kwargs,

    ) -> None:

        super().__init__()
        # TODO document this dropout as opposed to attention and activation dropouts
        self.dropout_module = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init

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

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

    def forward(
            self,
            batched_data,
    ):

        # x: B x T x C
        x, padding_mask, padded_index = self.graph_feature(batched_data)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)
        
        x = self._transformer_encoder(x, src_key_padding_mask=padding_mask)

        graph_rep = x[:, 0, :]

        return x, graph_rep