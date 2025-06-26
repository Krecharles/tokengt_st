#!/bin/bash

rm -r tokengt_experiments/pcqm4m/checkpoints

python -m tokengt_experiments.pcqm4m.pcqm4m_ogb_tokengt \
    --model token_gt \
    --checkpoint_dir tokengt_experiments/pcqm4m/checkpoints \
    --num_devices 1 \
    --lr 0.0002 \
    --epochs 50 \
    --warmup_epochs 5 \
    --weight_decay 0.01 \
    --batch_size 1024 \
    --D_P 64 \
    --head_dim 12 \
    --num_heads 24 \
    --num_encoder_layers 8 \
    --dim_feedforward 288 \
    --dropout_ratio 0.1 \
    --include_graph_token \
    --on_disk_dataset \
    --dataset_fraction 0.1
