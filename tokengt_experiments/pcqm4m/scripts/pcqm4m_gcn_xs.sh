#!/bin/bash

rm -r tokengt_experiments/pcqm4m/checkpoints

python -m tokengt_experiments.pcqm4m.pcqm4m_ogb_tokengt \
    --model gcn \
    --checkpoint_dir tokengt_experiments/pcqm4m/checkpoints \
    --save_test_dir tokengt_experiments/pcqm4m/test_dev \
    --num_devices 0 \
    --D_P 64 \
    --lr 0.001 \
    --epochs 30 \
    --warmup_epochs 2 \
    --weight_decay 0.001 \
    --batch_size 1024 \
    --hidden_channels 64 \
    --num_encoder_layers 8 \
    --dropout_ratio 0.3 \
    --on_disk_dataset \
    --dataset_fraction 0.01
