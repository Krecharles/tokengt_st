python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_no \
    --num_encoder_layers 4 \
    --hidden_channels 144 \
    --num_heads 8 \

# EMB -----------------------------

python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_yes \
    --use_mvn_no \
    --num_encoder_layers 4 \
    --hidden_channels 144 \
    --num_heads 8 \

python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config smarts-xl \
    --substructures_config none \
    --embed_smarts_yes \
    --use_mvn_no \
    --num_encoder_layers 4 \
    --hidden_channels 144 \
    --num_heads 8 \

python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_yes \
    --use_mvn_no \
    --num_encoder_layers 4 \
    --hidden_channels 144 \
    --num_heads 8 \

# MVN -----------------------------

python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 4 \
    --hidden_channels 144 \
    --num_heads 8 \

python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config smarts-xl \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 4 \
    --hidden_channels 144 \
    --num_heads 8 \

python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 4 \
    --hidden_channels 144 \
    --num_heads 8 \

# MVN +1 layer -----------------------------


python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_no \
    --num_encoder_layers 5 \
    --hidden_channels 144 \
    --num_heads 8 \

python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 5 \
    --hidden_channels 144 \
    --num_heads 8 \

python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config smarts-xl \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 5 \
    --hidden_channels 144 \
    --num_heads 8 \

python -m thesis_experiments.mpnn_exps \
    --architecture GAT \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 5 \
    --hidden_channels 144 \
    --num_heads 8 \
