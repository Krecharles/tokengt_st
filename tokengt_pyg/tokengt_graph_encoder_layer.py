"""
Modified from https://github.com/microsoft/Graphormer
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from .multihead_attention import MultiheadAttention
from .feedforward import FeedForward


class TokenGTGraphEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "gelu",
            init_fn: Callable = None,
            layernorm_style: str = "postnorm",
            return_attention: bool = False,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.layernorm_style = layernorm_style
        self.return_attention = return_attention

        self.dropout_module = nn.Dropout(dropout)

        # Initialize blocks
        self.self_attn = MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
            self_attention=True,
            )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.feedforward = self.build_FFN(
            self.embedding_dim,
            ffn_embedding_dim,
            activation_fn,
            activation_dropout,
            dropout,
            module_name=self.__class__.__name__
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def build_FFN(
            self,
            embedding_dim,
            ffn_embedding_dim,
            activation_fn,
            activation_dropout,
            dropout,
            module_name
    ):
        return FeedForward(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            activation_fn=activation_fn,
            activation_dropout=activation_dropout,
            dropout=dropout,
            module_name=module_name,
        )

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            self_attn_bias: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        if self.layernorm_style == "prenorm":
            residual = x
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                need_head_weights=self.return_attention,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.feedforward(x)
            x = residual + x

        elif self.layernorm_style == "postnorm":
            residual = x
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                need_head_weights=self.return_attention,
                attn_mask=self_attn_mask,
                attn_bias=self_attn_bias,
            )
            x = self.dropout_module(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.feedforward(x)
            x = residual + x
            x = self.final_layer_norm(x)

        else:
            raise NotImplementedError
        return x, attn
