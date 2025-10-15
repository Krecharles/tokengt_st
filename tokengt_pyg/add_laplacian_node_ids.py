from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms.base_transform import BaseTransform
import torch.nn.functional as F
import torch
from torch_geometric.utils._to_dense_adj import to_dense_adj
from torch_geometric.utils.laplacian import get_laplacian


@functional_transform("add_laplacian_node_identifiers_pyg")
class AddLaplacianNodeIdentifiers(BaseTransform):
    r"""Adds Laplacian node identifiers to a given input graph as described
    in the `"Pure Transformers are Powerful Graph Learners"
    <https://arxiv.org/pdf/2207.02505>`_ paper
    (functional name: :obj:`add_laplacian_node_identifiers`). Use as `pre_transform`
    to avoid unnecessary re-calculating of eigenvectors.

    Args:
        d_p (int): Dimension of node identifiers. If d_p is smaller than the
            number of nodes in the graph, the eigenvectors corresponding to
            the d_p smallest eigenvalues are used. If d_p is larger than the
            number of nodes in the graph, we zero pad channels.
    """

    def __init__(self, d_p: int):
        self._d_p = d_p

    def forward(self, data: Data) -> Data:
        assert data.num_nodes is not None
        assert data.edge_index is not None

        n = data.num_nodes
        lap_eigvec = self._get_lap_eigenvectors(data.edge_index, n)

        # Adapt lap_eigvec dimension to d_p
        lap_dim = lap_eigvec.size(-1)
        if self._d_p > lap_dim:
            lap_eigvec = F.pad(
                lap_eigvec, (0, self._d_p - lap_dim), value=float("0")
            )  # [sum(n_node), Dl]
        else:
            lap_eigvec = lap_eigvec[:, : self._d_p]  # [sum(n_node), Dl]

        data["node_ids"] = lap_eigvec
        return data

    @staticmethod
    def _get_lap_eigenvectors(edge_index: Tensor, n: int) -> Tensor:
        lap_edge_index, lap_edge_attr = get_laplacian(
            edge_index,
            normalization="sym",
            num_nodes=n,
        )
        lap_mat = to_dense_adj(lap_edge_index, edge_attr=lap_edge_attr)[0]
        _, eigenvectors = torch.linalg.eigh(lap_mat)

        return eigenvectors


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


class ConvertToSingleEmbTransform(BaseTransform):
    def __init__(self, offset: int = 512):
        self.offset = offset

    def forward(self, data: Data) -> Data:
        data.x = convert_to_single_emb(data.x, self.offset)
        return data
