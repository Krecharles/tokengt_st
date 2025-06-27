#!/bin/bash

python -m tokengt_experiments.pcqm4m.pcqm4m_ogb_tokengt \
    --model gat \
    --checkpoint_dir tokengt_experiments/pcqm4m/checkpoints_gat_paper \
    --num_devices 1 \
    --lr 0.0002 \
    --epochs 300 \
    --warmup_epochs 16 \
    --weight_decay 0.01 \
    --batch_size 1024 \
    --hidden_channels 600 \
    --num_heads 1 \
    --num_encoder_layers 5 \
    --dropout_ratio 0.1 \
    --on_disk_dataset \
    --dataset_fraction 1
