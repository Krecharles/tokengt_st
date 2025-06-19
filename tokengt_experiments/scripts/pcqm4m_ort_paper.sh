#!/bin/bash

python -m tokengt_experiments.pcqm4m_ogb_tokengt \
    --model token_gt \
    --log_dir pcqm4m/logs \
    --checkpoint_dir pcqm4m/checkpoints \
    --num_devices 1 \
    --lr 0.0002 \
    --iterations 1000000 \
    --warmup_iterations 60000 \
    --weight_decay 0.1 \
    --batch_size 1024 \
    --D_P 64 \
    --head_dim 12 \
    --num_heads 16 \
    --num_encoder_layers 6 \
    --dim_feedforward 192 \
    --dropout_ratio 0.1 \
    --include_graph_token \
    --on_disk_dataset
