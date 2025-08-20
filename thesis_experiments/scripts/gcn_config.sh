python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_no \
    --num_encoder_layers 4 \
    --hidden_channels 145 

# EMB -----------------------------

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_yes \
    --use_mvn_no \
    --num_encoder_layers 4 \
    --hidden_channels 145 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts-xl \
    --substructures_config none \
    --embed_smarts_yes \
    --use_mvn_no \
    --num_encoder_layers 4 \
    --hidden_channels 145 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_yes \
    --use_mvn_no \
    --num_encoder_layers 4 \
    --hidden_channels 145 

# MVN -----------------------------

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 4 \
    --hidden_channels 145 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts-xl \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 4 \
    --hidden_channels 145 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 4 \
    --hidden_channels 145 

# MVN +1 layer -----------------------------

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_no \
    --num_encoder_layers 5 \
    --hidden_channels 145 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 5 \
    --hidden_channels 145 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts-xl \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 5 \
    --hidden_channels 145 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 5 \
    --hidden_channels 145 

# xs -----------------------------
python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_no \
    --num_encoder_layers 6 \
    --hidden_channels 8 


# EMB-xs -----------------------------

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_yes \
    --use_mvn_no \
    --num_encoder_layers 6 \
    --hidden_channels 8 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts-xl \
    --substructures_config none \
    --embed_smarts_yes \
    --use_mvn_no \
    --num_encoder_layers 6 \
    --hidden_channels 8 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_yes \
    --use_mvn_no \
    --num_encoder_layers 6 \
    --hidden_channels 8 


# MVN-xs -----------------------------

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 6 \
    --hidden_channels 8 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config smarts-xl \
    --substructures_config none \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 6 \
    --hidden_channels 8 

python3 -m thesis_experiments.mpnn_exps \
    --architecture GCN \
    --smarts_config none \
    --substructures_config cycles \
    --embed_smarts_no \
    --use_mvn_yes \
    --num_encoder_layers 6 \
    --hidden_channels 8 