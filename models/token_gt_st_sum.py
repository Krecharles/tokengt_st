from typing import List, Optional, Tuple

from torch_geometric.utils._unbatch import unbatch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import TokenGT


class TokenGTST_Sum(TokenGT):
    def __init__(
        self,
        *args,
        n_substructures: int,
        **transformer_kwargs
    ):
        super().__init__(*args, **transformer_kwargs)
        self.n_substructures = n_substructures

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
        substructure_instances: List[List[List[List[int]]]],
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
            substructure_instances (List[List[List[List[int]]]]): 
                [batch_size, num_substrucs, num_instances, num_instance_nodes]
        """
        batched_emb, src_key_padding_mask, node_mask = (
            self._get_tokenwise_batched_emb(x, edge_index, edge_attr, ptr,
                                            batch, node_ids, substructure_instances))
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

    def _get_tokenwise_batched_emb(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
        node_ids: Tensor,
        substructure_instances: List[List[List[List[int]]]],
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
            substructure_instances, ptr, node_ids)

        # combine node + edge tokens,
        # and split graphs into padded batches -> [batch_size, max_tokens, d]
        n_nodes = ptr[1:] - ptr[:-1]
        n_edges = self._get_n_edges(edge_index, ptr)

        # n_substruct_tokens = [(len(instances) for instances in b) for b in len(substructure_instances)]
        n_substruct_tokens = []
        for b in substructure_instances:
            n_substruct_tokens.append(
                sum([len(substructs) for substructs in b]))
        n_substruct_tokens = torch.tensor(n_substruct_tokens)

        ptr_substructs = torch.cat(
            [torch.tensor([0]), torch.cumsum(n_substruct_tokens, dim=0)])

        n_tokens = n_nodes + n_edges + n_substruct_tokens
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
        substructure_instances: List[List[List[List[int]]]],
        ptr: Tensor,
        node_ids: Tensor,
    ) -> Tensor:
        r"""adds projected node identifiers of all substructure involved nodes and 
        the type identifiers for the respective substructure

        edge token embedding = node_ids_prj(sum node_ids) + type_ids for the substructure 

        Args:
            substructure_instances (List[List[List[List[int]]]]): 
                [batch_size, num_substrucs, num_instances, num_instance_nodes]
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

        # For every graph in the batch batch:
        # For every substructure, get its type id
        # Then for every of its substructure instances, get the nodes
        # For all of these nodes, offset them wrt the batch and sum the node_ids

        structural_tokens = []
        for b in range(len(substructure_instances)):
            for substruc_idx, substruc in enumerate(substructure_instances[b]):
                type_id = self._type_id_enc.weight[2 + substruc_idx]
                for instance_nodes in substruc:
                    # Adapt the instance_nodes to their offset in the batch.
                    sub_nodes_sum = node_ids[torch.tensor(instance_nodes) +
                                             ptr[b]].sum(dim=0)
                    node_ids_prj = self._node_id_enc(
                        torch.concat((sub_nodes_sum, sub_nodes_sum), 0))
                    structural_tokens.append(node_ids_prj + type_id)

        if len(structural_tokens) == 0:
            return torch.empty((0, self._d), device=self._device)
        else:
            # [n_substructure_tokens, d]
            return torch.stack(structural_tokens, dim=0)

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
            graph_substructure_emb = substructure_emb[ptr_substructs[i]
                :ptr_substructs[i + 1]]
            unpadded_emb = torch.concat(
                (graph_node_emb, graph_edge_emb, graph_substructure_emb), 0)
            pad = (0, 0, 0, max_tokens - n_tokens[i])
            padded_emb = F.pad(unpadded_emb, pad, value=0.0).unsqueeze(0)
            batched_emb.append(padded_emb)
        return torch.concat(batched_emb, 0)  # [b, t, c]
