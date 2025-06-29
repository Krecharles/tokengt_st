from typing import List
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils.convert import to_networkx


@functional_transform("add_substructure_instances")
class AddSubstructureInstances(BaseTransform):

    def __init__(self, substructures: List[nx.Graph]):
        self._substructures = substructures
        if len(substructures) == 0:
            self._largest_substructure_size = 0
        else:
            self._largest_substructure_size = max(substructure.number_of_nodes() for substructure in substructures)

    def forward(self, data: Data) -> Data:
        assert data.num_nodes is not None
        assert data.edge_index is not None

        G = to_networkx(data, to_undirected=True)

        substructure_instances = []
        for i, gpattern in enumerate(self._substructures):
            instances = self._find_uniques(G, gpattern)
            # prepend substructure type id to each instance and match largest substructure size
            instances = [[i] + instance + [-1] * (self._largest_substructure_size - len(instance)) for instance in instances]
            substructure_instances.extend(instances)

        data["substructure_instances"] = torch.tensor(substructure_instances, dtype=torch.long)
        data["n_substructure_instances"] = torch.tensor(len(substructure_instances), dtype=torch.long)
        return data

    @staticmethod
    def _find_uniques(G, gpattern):
        unique_subgraphs = []
        out = []
        matcher = nx.algorithms.isomorphism.GraphMatcher(G, gpattern)
        for mapping in matcher.subgraph_isomorphisms_iter():
            # Convert mapping values to a frozenset of node IDs (order doesn't matter)
            subgraph_nodes = frozenset(mapping.keys())
            if subgraph_nodes not in unique_subgraphs:
                unique_subgraphs.append(frozenset(mapping.keys()))
                out.append(list(mapping.keys()))
        return out
