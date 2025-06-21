#!/bin/bash

rm -r pcqm4m

python -m tokengt_experiments.pcqm4m_ogb_tokengt \
    --model gcn \
    --checkpoint_dir pcqm4m/checkpoints \
    --num_devices 1 \
    --lr 0.0002 \
    --epochs 30 \
    --warmup_epochs 2 \
    --weight_decay 0.01 \
    --batch_size 1024 \
    --D_P 64 \
    --head_dim 12 \
    --num_heads 16 \
    --num_encoder_layers 6 \
    --dim_feedforward 192 \
    --dropout_ratio 0.1 \
    --include_graph_token \
    --on_disk_dataset \
    --dataset_fraction 0.01 
