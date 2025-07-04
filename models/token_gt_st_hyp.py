from typing import List, Optional, Tuple

from torch_geometric.utils._unbatch import unbatch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import TokenGT


class TokenGTST_Hyp(TokenGT):
    def __init__(
        self,
        *args,
        n_substructures: int,
        higher_order_dim: int = 6,
        **transformer_kwargs
    ):
        super().__init__(*args, **transformer_kwargs)

        assert higher_order_dim % 2 == 0

        self.n_substructures = n_substructures
        self.higher_order_dim = higher_order_dim

        self._node_id_enc = nn.Linear(
            self._d_p * higher_order_dim, self._d, False, self._device)

        # Reassign the _type_id_enc to accomodate for the new type ids.
        self._type_id_enc = nn.Embedding(
            2+n_substructures, self._d, device=self._device)

        # re-initialise parameters
        self.apply(lambda m: self._init_params(m, self._num_encoder_layers))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
        node_ids: Tensor,
        substructure_instances: Tensor,
        n_substructure_instances: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Forward pass that returns embeddings for each input node and
        (optionally) a graph-level embedding for each graph in the input.

        Args:
            x (torch.Tensor): The input node features. Needs to have number of
                channels equal to dim_node.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features. If provided,
                needs to have number of channels equal to dim_edge.
            ptr (torch.Tensor): The pointer vector that provides a cumulative
                sum of each graph's node count. The number of entries is one
                more than the number of input graphs. Note: when providing a
                single graph with (say) 5 nodes as input, set equal to
                torch.tensor([0, 5]).
            batch (torch.Tensor): The batch vector that relates each node to a
                specific graph. The number of entries is equal to the number of
                rows in x. Note: when providing a single graph with (say) 5
                nodes as input, set equal to torch.tensor([0, 0, 0, 0, 0]).
            node_ids (torch.Tensor): Orthonormal node identifiers (needs to
                have number of channels equal to d_p).
                A map of n_substructures entries, 
            substructure_instances (Tensor): 
                [num_substrucs, num_instances, num_instance_nodes]
            n_substructure_instances (Tensor):
                [batch_size]
        """
        batched_emb, src_key_padding_mask, node_mask = (
            self._get_tokenwise_batched_emb(x, edge_index, edge_attr, ptr,
                                            batch, node_ids, substructure_instances, n_substructure_instances))
        if self._graph_emb is not None:
            # append special graph token
            b_s = batched_emb.shape[0]
            graph_emb = self._graph_emb.weight.expand(b_s, 1, -1)
            batched_emb = torch.concat((graph_emb, batched_emb), 1)
            b_t = torch.tensor([False], device=self._device).expand(b_s, -1)
            src_key_padding_mask = torch.concat((b_t, src_key_padding_mask), 1)

        batched_emb = self._encoder(batched_emb, None, src_key_padding_mask)
        if self._graph_emb is not None:
            # grab graph token embedding from each batch
            graph_emb = batched_emb[:, 0, :]
            batched_emb = batched_emb[:, 1:, :]
        else:
            graph_emb = None

        # each batch has node + edge + padded entries;
        # select node emb and collapse into 2d tensor that matches x
        node_emb = batched_emb[node_mask]
        return node_emb, graph_emb

    def _get_node_token_emb(self, x: Tensor, node_ids: Tensor) -> Tensor:
        r"""Concats the node identifiers higher_order_dim times and applies 
        linear projection to node features, and adds projected node
        identifiers and type identifiers.

        node token embedding = x_prj + node_ids_prj + type_ids
        """
        x_prj = self._node_features_enc(x)
        # Only line changed. repeat every node_id multiple times
        node_ids_prj = self._node_id_enc(
            node_ids.repeat(1, self.higher_order_dim))
        total_nodes = x.shape[0]
        type_ids = self._type_id_enc.weight[0].expand(total_nodes, -1)

        node_emb = x_prj + node_ids_prj + type_ids
        return node_emb  # [total_nodes, d]

    def _get_edge_token_emb(
        self,
        edge_attr: Optional[Tensor],
        edge_index: Tensor,
        node_ids: Tensor,
    ) -> Tensor:
        r"""For edge (u,v), concats the node identifiers of u higher_order_dim/2 times 
        with the  the node identifiers of v higher_order_dim/2 times. 
        Applies linear projection to edge features (if present), and adds
        projected node identifiers and type identifiers.

        edge token embedding = edge_attr_prj + node_ids_prj + type_ids
        """
        if edge_attr is not None:
            edge_attr_prj = self._edge_features_enc(edge_attr)
        else:
            edge_attr_prj = None

        src_emb = node_ids[edge_index[0]].repeat(1, self.higher_order_dim//2)
        tar_emb = node_ids[edge_index[1]].repeat(1, self.higher_order_dim//2)
        node_ids_concat = torch.concat((src_emb, tar_emb), 1)
        node_ids_prj = self._node_id_enc(node_ids_concat)
        total_edges = edge_index.shape[1]
        type_ids = self._type_id_enc.weight[1].expand(total_edges, -1)

        edge_emb = node_ids_prj + type_ids
        if edge_attr_prj is not None:
            edge_emb = edge_emb + edge_attr_prj
        return edge_emb  # [total_edges, d]

    def _get_tokenwise_batched_emb(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
        node_ids: Tensor,
        substructure_instances: Tensor,
        n_substructure_instances: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Adds all identifiers, and batches data due to different
        graphs. Returns batched tokenized embeddings, together with masks.
        """
        if self._is_laplacian_node_ids is True and self.training is True:
            # flip eigenvector signs and apply dropout
            unbatched_node_ids = list(unbatch(node_ids, batch))
            for i in range(len(unbatched_node_ids)):
                sign = -1 + 2 * torch.randint(0, 2, (self._d_p, ),
                                              device=self._device)
                unbatched_node_ids[i] = sign * unbatched_node_ids[i]
                unbatched_node_ids[i] = self._node_id_dropout(
                    unbatched_node_ids[i])
            node_ids = torch.concat(unbatched_node_ids, 0)

        node_emb = self._get_node_token_emb(x, node_ids)
        edge_emb = self._get_edge_token_emb(edge_attr, edge_index, node_ids)
        substructure_emb = self._get_substructure_token_emb(
            substructure_instances, n_substructure_instances, ptr, node_ids)

        # combine node + edge tokens,
        # and split graphs into padded batches -> [batch_size, max_tokens, d]
        n_nodes = ptr[1:] - ptr[:-1]
        n_edges = self._get_n_edges(edge_index, ptr)

        ptr_substructs = torch.cat(
            [torch.tensor([0], device=self._device), torch.cumsum(n_substructure_instances, dim=0)])

        n_tokens = n_nodes + n_edges + n_substructure_instances
        batched_emb = self._get_batched_emb(
            node_emb,
            edge_emb,
            substructure_emb,
            ptr,
            edge_index,
            n_tokens,
            ptr_substructs
        )

        # construct self-attention and node masks
        src_key_padding_mask = self._get_src_key_padding_mask(n_tokens)
        node_mask = self._get_node_mask(n_tokens, n_nodes)

        return batched_emb, src_key_padding_mask, node_mask

    def _get_substructure_token_emb(
        self,
        substructure_instances: Tensor,
        n_substructure_instances: Tensor,
        ptr: Tensor,
        node_ids: Tensor,
    ) -> Tensor:
        r"""adds projected node identifiers of all substructure involved nodes and 
        the type identifiers for the respective substructure

        edge token embedding = node_ids_prj + type_ids for the substructure 

        Args:
            substructure_instances (Tensor): 
                [num_substrucs, num_instances, num_instance_nodes]
            n_substructure_instances (Tensor):
                [batch_size]
            ptr (torch.Tensor): The pointer vector that provides a cumulative
                sum of each graph's node count. The number of entries is one
                more than the number of input graphs. Note: when providing a
                single graph with (say) 5 nodes as input, set equal to
                torch.tensor([0, 5]).
            batch (torch.Tensor): The batch vector that relates each node to a
                specific graph. The number of entries is equal to the number of
                rows in x. Note: when providing a single graph with (say) 5
                nodes as input, set equal to torch.tensor([0, 0, 0, 0, 0]).
            node_ids (torch.Tensor): Orthonormal node identifiers (needs to
                have number of channels equal to d_p).
        """

        # TODO implement general case
        assert self.higher_order_dim == 6
        assert not torch.any(substructure_instances == -1)

        # For every graph in the batch batch:
        # For every substructure, get its type id
        # Then for every of its substructure instances, get the nodes
        # For all of these nodes, offset them wrt the batch and concat the node_ids

        keys = substructure_instances[:, 0] # [num_substrucs]
        vertices = substructure_instances[:, 1:] # [num_substr, num_vertices]

        # Offset vertices because they are batched.
        instance_to_graph = torch.arange(len(n_substructure_instances), device=self._device).repeat_interleave(n_substructure_instances)
        offsets = ptr[instance_to_graph].unsqueeze(1)

        offset_vertices = vertices.clone() + offsets

        # reshape node_ids_prj from [num_instances, num_instance_nodes, d_p] to [num_substrucs, num_instances * d_p]
        substrs_node_ids = node_ids[offset_vertices].reshape(substructure_instances.shape[0], -1)

        node_ids_prj = self._node_id_enc(substrs_node_ids)
        type_ids = self._type_id_enc.weight[2 + keys]

        return type_ids + node_ids_prj

    @staticmethod
    @torch.no_grad()
    def _get_batched_emb(
        node_emb: Tensor,
        edge_emb: Tensor,
        substructure_emb: Tensor,
        ptr: Tensor,
        edge_index: Tensor,
        n_tokens: Tensor,
        ptr_substructs: Tensor
    ) -> Tensor:
        r"""Combines node and edge embeddings of each input graph, and pads the
        time dimension to equal that of the input graph with the most nodes +
        edges.
        """
        max_tokens = n_tokens.max().item()
        batch_size = n_tokens.shape[0]
        batched_emb = []

        for i in range(batch_size):
            graph_node_emb = node_emb[ptr[i]:ptr[i + 1]]
            graph_edge_emb = edge_emb[(edge_index[0] >= ptr[i])
                                      & (edge_index[0] < ptr[i + 1])]
            graph_substructure_emb = substructure_emb[ptr_substructs[i]                                                      :ptr_substructs[i + 1]]
            unpadded_emb = torch.concat(
                (graph_node_emb, graph_edge_emb, graph_substructure_emb), 0)
            pad = (0, 0, 0, max_tokens - n_tokens[i])
            padded_emb = F.pad(unpadded_emb, pad, value=0.0).unsqueeze(0)
            batched_emb.append(padded_emb)
        return torch.concat(batched_emb, 0)  # [b, t, c]
