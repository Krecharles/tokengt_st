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
            encoder_normalize_before: bool = False,
            layernorm_style: str = "postnorm",
            apply_graphormer_init: bool = False,
            activation_fn: str = "gelu",

            return_attention: bool = False,

    ) -> None:

        super().__init__()
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

        if layernorm_style == "prenorm":
            self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.layers = nn.ModuleList([])

        self.layers.extend(
            [
                self.build_tokengt_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    layernorm_style=layernorm_style,
                    return_attention=return_attention,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

    def build_tokengt_graph_encoder_layer(
            self,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
            layernorm_style,
            return_attention,
    ):
        return TokenGTGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            layernorm_style=layernorm_style,
            return_attention=return_attention
        )

    def forward(
            self,
            batched_data,
            perturb=None,
            last_state_only: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        if token_embeddings is not None:
            raise NotImplementedError
        else:
            x, padding_mask, padded_index = self.graph_feature(batched_data, perturb)

        # x: B x T x C

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        if attn_mask is not None:
            raise NotImplementedError


        attn_dict = {'maps': {}, 'padded_index': padded_index}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, attn = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask)
            if not last_state_only:
                inner_states.append(x)
            attn_dict['maps'][i] = attn

        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        return inner_states, graph_rep, attn_dict