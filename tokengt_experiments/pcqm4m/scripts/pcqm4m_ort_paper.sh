#!/bin/bash

python -m tokengt_experiments.pcqm4m.pcqm4m_ogb_tokengt \
    --model token_gt \
    --checkpoint_dir tokengt_experiments/pcqm4m/checkpoints_ort_paper \
    --num_devices 1 \
    --num_workers 16 \
    --lr 0.0002 \
    --warmup_iterations 12000 \
    --iterations 200000 \
    --weight_decay 0.1 \
    --batch_size 512 \
    --D_P 64 \
    --node_id_mode orf \
    --head_dim 24 \
    --num_heads 32 \
    --num_encoder_layers 12 \
    --dim_feedforward 768 \
    --dropout_ratio 0.1 \
    --include_graph_token \
    --dataset_fraction 1 \
    --on_disk_dataset \
    --fp16
