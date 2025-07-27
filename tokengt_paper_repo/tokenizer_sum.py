import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import GraphFeatureTokenizer, init_params
from .orf import gaussian_orthogonal_random_matrix_batched



class GraphFeatureTokenizerSum(GraphFeatureTokenizer):
    """
    Compute node and edge features for each node and edge in the graph.
    """

    def __init__(
            self,
            n_substructures: int,
            *args,
            **kwargs,
    ):
        super(GraphFeatureTokenizerSum, self).__init__(*args, **kwargs)

        if self.type_id:
            self.substructure_type_encoder = nn.Embedding(n_substructures, self.encoder_embed_dim)

        self.apply(lambda module: init_params(module, n_layers=self.n_layers))

    @staticmethod
    def get_batch(node_feature, edge_index, edge_feature, node_num, edge_num, n_substructures_instances, perturb=None):
        """
        :param node_feature: Tensor([sum(node_num), D])
        :param edge_index: LongTensor([2, sum(edge_num)])
        :param edge_feature: Tensor([sum(edge_num), D])
        :param node_num: list
        :param edge_num: list
        :param n_substructures_instances: Tensor([B])
        :param perturb: Tensor([B, max(node_num), D])
        :return: padded_index: LongTensor([B, T, 2]), padded_feature: Tensor([B, T, D]), padding_mask: BoolTensor([B, T])
        """
        seq_len = [n + e + s for n, e, s in zip(node_num, edge_num, n_substructures_instances)]
        b = len(seq_len)
        d = node_feature.size(-1)
        max_len = max(seq_len)
        max_n = max(node_num)
        device = edge_index.device

        token_pos = torch.arange(max_len, device=device)[None, :].expand(b, max_len)  # [B, T]

        seq_len = torch.tensor(seq_len, device=device, dtype=torch.long)[:, None]  # [B, 1]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None]  # [B, 1]
        edge_num = torch.tensor(edge_num, device=device, dtype=torch.long)[:, None]  # [B, 1]

        node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(b, max_n)  # [B, max_n]
        node_index = node_index[None, node_index < node_num].repeat(2, 1)  # [2, sum(node_num)]

        padded_node_mask = torch.less(token_pos, node_num)
        padded_edge_mask = torch.logical_and(
            torch.greater_equal(token_pos, node_num),
            torch.less(token_pos, node_num + edge_num)
        )
        padded_token_mask = torch.logical_and(
            torch.greater_equal(token_pos, node_num + edge_num),
            torch.less(token_pos, seq_len)
        )

        # padded index is a mapping of tokens to (u, u) or (u, v)
        padded_index = torch.zeros(b, max_len, 2, device=device, dtype=torch.long)  # [B, T, 2]
        padded_index[padded_node_mask, :] = node_index.t()
        padded_index[padded_edge_mask, :] = edge_index.t()

        if perturb is not None:
            perturb_mask = padded_node_mask[:, :max_n]  # [B, max_n]
            node_feature = node_feature + perturb[perturb_mask].type(node_feature.dtype)  # [sum(node_num), D]

        padded_feature = torch.zeros(b, max_len, d, device=device, dtype=node_feature.dtype)  # [B, T, D]
        padded_feature[padded_node_mask, :] = node_feature
        padded_feature[padded_edge_mask, :] = edge_feature

        padding_mask = torch.greater_equal(token_pos, seq_len)  # [B, T]
        return padded_index, padded_feature, padding_mask, padded_node_mask, padded_edge_mask, padded_token_mask

    @staticmethod
    def get_index_embed(node_id, node_mask, padded_index):
        """
        :param node_id: Tensor([sum(node_num), D])
        :param node_mask: BoolTensor([B, max_n])
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, 2D])
        """
        b, max_n = node_mask.size()
        max_len = padded_index.size(1)
        d = node_id.size(-1)

        padded_node_id = torch.zeros(b, max_n, d, device=node_id.device, dtype=node_id.dtype)  # [B, max_n, D]
        padded_node_id[node_mask] = node_id

        padded_node_id = padded_node_id[:, :, None, :].expand(b, max_n, 2, d)
        padded_index = padded_index[..., None].expand(b, max_len, 2, d)
        index_embed = padded_node_id.gather(1, padded_index)  # [B, T, 2, D]
        index_embed = index_embed.view(b, max_len, 2 * d)
        return index_embed
    
    def get_type_embed(self, padded_index):
        """
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, D])
        """
        order = torch.eq(padded_index[..., 0], padded_index[..., 1]).long()  # [B, T]
        order_embed = self.order_encoder(order)
        return order_embed

    @staticmethod
    def get_substructure_token_embed(node_id, node_mask, substructure_instances, n_substructures_instances):
        """
        :param node_id: Tensor([sum(node_num), D])
        :param node_mask: BoolTensor([B, max_n])
        :param substructure_instances: Tensor([n_substructures, S])
        :param n_substructures_instances: Tensor([B])
        :return: keys: Tensor([S]), substr_index_sum: Tensor([S, D])
        """

        b, max_n = node_mask.size()
        d = node_id.size(-1)

        padded_node_id = torch.zeros(b, max_n, d, device=node_id.device, dtype=node_id.dtype)  # [B, max_n, D]
        padded_node_id[node_mask] = node_id

        keys = substructure_instances[:, 0] # [num_substrucs]
        vertices = substructure_instances[:, 1:] # [num_substr, num_vertices]
        mask = vertices != -1
        vertices[~mask] = 0

        batch_ids = torch.arange(b, device=node_id.device).repeat_interleave(n_substructures_instances).unsqueeze(1)
        substr_node_id = padded_node_id[batch_ids, vertices] # [num_substrucs, num_vertices, D]
        substr_node_id = substr_node_id * mask.unsqueeze(-1)
        substr_index_sum = substr_node_id.sum(1) # [num_substrucs, D]

        return keys, substr_index_sum

    def forward(self, batched_data, perturb=None):
        (
            node_data,
            batch,
            ptr,
            eigvec,
            edge_index,
            edge_data,
            substructure_instances,
            n_substructures_instances,
        ) = (
            batched_data["node_data"],
            batched_data["batch"],
            batched_data["ptr"],
            batched_data["lap_eigvec"],
            batched_data["edge_index"],
            batched_data["edge_data"],
            batched_data["substructure_instances"],
            batched_data["n_substructure_instances"],
        )

        node_num = ptr[1:] - ptr[:-1]
        edge_num = torch.bincount(batch[edge_index[0]], minlength=int(batch.max()) + 1)

        # remove batchting offsets from edge_index
        edge_index = edge_index - torch.repeat_interleave(ptr[:-1], edge_num).unsqueeze(0)

        node_feature = self.atom_encoder(node_data.int()).sum(-2)  # [sum(n_node), D]
        edge_feature = self.edge_encoder(edge_data.int()).sum(-2)  # [sum(n_edge), D]
        device = node_feature.device
        dtype = node_feature.dtype

        padded_index, padded_feature, padding_mask, _, _, padded_token_mask = self.get_batch(
            node_feature, edge_index, edge_feature, node_num, edge_num, n_substructures_instances, perturb
        )
        node_mask = self.get_node_mask(node_num, node_feature.device)  # [B, max(n_node)]

        if self.rand_node_id:
            rand_node_id = torch.rand(sum(node_num), self.rand_node_id_dim, device=device, dtype=dtype)  # [sum(n_node), D]
            rand_node_id = F.normalize(rand_node_id, p=2, dim=1)
            rand_index_embed = self.get_index_embed(rand_node_id, node_mask, padded_index)  # [B, T, 2D]
            padded_feature = padded_feature + self.rand_encoder(rand_index_embed)
            keys, substr_index_sum = self.get_substructure_token_embed(
                rand_node_id, node_mask, substructure_instances, n_substructures_instances)

        if self.orf_node_id:
            b, max_n = len(node_num), max(node_num)
            orf = gaussian_orthogonal_random_matrix_batched(
                b, max_n, max_n, device=device, dtype=dtype
            )  # [b, max(n_node), max(n_node)]
            orf_node_id = orf[node_mask]  # [sum(n_node), max(n_node)]
            if self.orf_node_id_dim > max_n:
                orf_node_id = F.pad(orf_node_id, (0, self.orf_node_id_dim - max_n), value=float('0'))  # [sum(n_node), Do]
            else:
                orf_node_id = orf_node_id[..., :self.orf_node_id_dim]  # [sum(n_node), Do]
            orf_node_id = F.normalize(orf_node_id, p=2, dim=1)
            orf_index_embed = self.get_index_embed(orf_node_id, node_mask, padded_index)  # [B, T, 2Do]
            padded_feature = padded_feature + self.orf_encoder(orf_index_embed)
            keys, substr_index_sum = self.get_substructure_token_embed(
                orf_node_id, node_mask, substructure_instances, n_substructures_instances)


        if self.lap_node_id:
            # lap_dim = lap_eigvec.size(-1)
            # if self.lap_node_id_k > lap_dim:
            #     eigvec = F.pad(lap_eigvec, (0, self.lap_node_id_k - lap_dim), value=float('0'))  # [sum(n_node), Dl]
            # else:
            #     eigvec = lap_eigvec[:, :self.lap_node_id_k]  # [sum(n_node), Dl]
            if self.lap_eig_dropout is not None:
                eigvec = self.lap_eig_dropout(eigvec[..., None, None]).view(eigvec.size())
            lap_node_id = self.handle_eigvec(eigvec, node_mask, self.lap_node_id_sign_flip)
            lap_index_embed = self.get_index_embed(lap_node_id, node_mask, padded_index)  # [B, T, 2Dl]
            padded_feature = padded_feature + self.lap_encoder(lap_index_embed)
            keys, substr_index_sum = self.get_substructure_token_embed(
                lap_node_id, node_mask, substructure_instances, n_substructures_instances)

        if self.type_id:
            padded_feature = padded_feature + self.get_type_embed(padded_index)

        substructure_type_embed = self.substructure_type_encoder(keys)
        substructure_index_embed = torch.cat([substr_index_sum, substr_index_sum], dim=1)
        padded_feature[padded_token_mask] = substructure_type_embed + self.lap_encoder(substructure_index_embed)

        padded_feature, padding_mask = self.add_special_tokens(padded_feature, padding_mask)  # [B, 2+T, D], [B, 2+T]

        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float('0'))
        return padded_feature, padding_mask, padded_index  # [B, 2+T, D], [B, 2+T], [B, T, 2]
