#!/bin/bash

python -m tokengt_experiments.pcqm4m_ogb_tokengt \
    --model token_gt \
    --log_dir pcqm4m/logs \
    --num_devices 0 \
    --initial_lr 0.001 \
    --lr_reduce_factor 0.5 \
    --minimum_lr 1e-5 \
    --patience 10 \
    --batch_size 256 \
    --epochs 1000 \
    --D_P 64 \
    --head_dim 16 \
    --num_heads 4 \
    --num_encoder_layers 2 \
    --dim_feedforward 128 \
    --dropout_ratio 0.1 \
    --include_graph_token
