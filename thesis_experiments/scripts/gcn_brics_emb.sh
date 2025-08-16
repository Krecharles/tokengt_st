python -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config none \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_no \
    --num_encoder_layers 8 \
    --hidden_channels 64
    
python -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config brics \
    --substructures_config none \
    --embed_smarts_yes \
    --use_mvn_no

python -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config brics \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \

python -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_yes \
    --use_mvn_no

python -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_no \
    --use_mvn_yes \
