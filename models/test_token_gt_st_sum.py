import pytest
import torch

from models.token_gt_st_sum import TokenGTST_Sum


@pytest.mark.parametrize("dim_edge", [None, 16])
@pytest.mark.parametrize("include_graph_token", [True, False])
@pytest.mark.parametrize("is_laplacian_node_ids", [True, False])
@pytest.mark.parametrize("is_multiple_graph_input", [True, False])
@pytest.mark.parametrize("no_substructure_instances", [True, False])
def test_token_gt_st(
    dim_edge,
    include_graph_token,
    is_laplacian_node_ids,
    is_multiple_graph_input,
    no_substructure_instances,
):
    dim_node, d_p = 10, 5
    x = torch.rand(5, dim_node)
    # An edge and a triangle.
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 2, 4, 3, 4], [1, 0, 3, 2, 4, 2, 4, 3]])
    edge_attr = torch.rand(8, dim_edge) if dim_edge is not None else None
    if is_multiple_graph_input:
        ptr = torch.tensor([0, 2, 5])
        batch = torch.tensor([0, 0, 1, 1, 1])
        # 1 batch with 1 triangle with 0 instances and 1 batch with 1 trinagle with 1 instance of nodes 2, 3, 4
        if no_substructure_instances:
            substructure_instances = [[[]], [[]]]
        else:
            substructure_instances = [[[]], [[[0, 1, 2]]]]
    else:
        ptr = torch.tensor([0, 5])
        batch = torch.tensor([0, 0, 0, 0, 0])
        if no_substructure_instances:
            substructure_instances = [[[]]]
        else:
            # 1 batch with 1 triangle with 1 instance of nodes 2, 3, 4
            substructure_instances = [[[[2, 3, 4]]]]
    node_ids = torch.rand(5, d_p)

    model = TokenGTST_Sum(
        dim_node=10,
        dim_edge=dim_edge,
        d_p=d_p,
        d=16,
        num_heads=1,
        num_encoder_layers=1,
        dim_feedforward=16,
        include_graph_token=include_graph_token,
        is_laplacian_node_ids=is_laplacian_node_ids,
        n_substructures=1  # Just a triangle
    )
    model.reset_params()
    assert str(model) == "TokenGTST_Sum(16)"

    node_emb, graph_emb = model(
        x, edge_index, edge_attr, ptr, batch, node_ids, substructure_instances)
    assert node_emb.shape == (5, 16)
    if include_graph_token and is_multiple_graph_input:
        assert graph_emb.shape == (2, 16)
    elif include_graph_token and not is_multiple_graph_input:
        assert graph_emb.shape == (1, 16)
    else:
        assert graph_emb is None
