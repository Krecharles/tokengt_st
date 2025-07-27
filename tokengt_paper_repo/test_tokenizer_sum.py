import torch


from tokengt_paper_repo.tokenizer_sum import GraphFeatureTokenizerSum

def test_get_substructure_token_embed():
    d_p = 3
    node_num = torch.tensor([3, 4])
    node_mask = GraphFeatureTokenizerSum.get_node_mask(node_num, device="cpu")
    substructure_instances = torch.tensor([[0, 0, 1, -1], [0, 1, 2, -1], [1, 0, 1, 2]])
    n_substructures_instances = torch.tensor([2, 1])

    node_id = torch.randn(sum(node_num), d_p)

    keys, index_sum_embed = GraphFeatureTokenizerSum.get_substructure_token_embed(
        node_id,
        node_mask,
        substructure_instances,
        n_substructures_instances,
    )

    assert keys.shape == (3,)
    assert index_sum_embed.shape == (3, d_p)
    assert torch.all(keys == torch.tensor([0, 0, 1]))
    assert torch.all(index_sum_embed[0] == node_id[0] + node_id[1])
    assert torch.all(index_sum_embed[1] == node_id[1] + node_id[2])
    assert torch.all(index_sum_embed[2] == node_id[3] + node_id[4] + node_id[5])