import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from torch_geometric.data import Data
from add_substructure_instances import AddSubstructureInstances
import networkx as nx


def create_test_graph():
    # Create a triangle graph (nodes: 0-1-2-0)
    edge_index = torch.tensor([[0, 1, 2],
                               [1, 2, 0]], dtype=torch.long)
    x = torch.ones((3, 1))
    return Data(x=x, edge_index=edge_index)


def create_triangle_subgraph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G


def create_3path_subgraph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    return G


def test_forward_triangle_match():
    triangle = create_triangle_subgraph()
    data = create_test_graph()
    transform = AddSubstructureInstances([triangle])

    result = transform(data)
    instances = result['substructure_instances']
    assert len(instances) == 1
    assert len(instances[0]) == 1  # One triangle instance
    assert set(instances[0][0]) == {0, 1, 2}


def test_forward_no_match():
    path = nx.Graph()
    # longer path than in test graph
    path.add_edges_from([(0, 1), (1, 2), (2, 3)])
    data = create_test_graph()
    transform = AddSubstructureInstances([path])

    result = transform(data)
    instances = result['substructure_instances']
    assert len(instances) == 1
    assert instances[0] == []  # No matches found


def test_multiple_substructures():
    triangle = create_triangle_subgraph()
    path = create_3path_subgraph()
    data = create_test_graph()
    transform = AddSubstructureInstances([triangle, path])

    result = transform(data)
    instances = result['substructure_instances']
    assert len(instances) == 2
    # Triangle match
    assert len(instances[0]) == 1
    assert set(instances[0][0]) == {0, 1, 2}

    # Path matches (should be 1 unique set of nodes)
    # TODO the problem is that the 3-path has no isomorphism to the triangle
    # because the 3rd edge would be induced as well. So isomorphism counting !=
    # subgraph counting. Discuss if this is important
    assert len(instances[1]) == 1
    assert set(instances[1][0]) == {0, 1, 2}
