# TokenGT with Structural Tokens

For setup, clone the [PyG TokenGT PR](https://github.com/michailmelonas/pytorch_geometric) and put the `pytorch_geomtric` folder in the root folder of this project.

commands to setup the pyg fork:

```
git clone https://github.com/Krecharles/tokengt_st
git clone https://github.com/krecharles/pytorch_geometric.git
cd pytorch_geometric/
git checkout add-token-gt
cd ..
pip install -e pytorch_geometric
cd tokengt_st
pip install -r requirements.txt
```

touch ~/.no_auto_tmux
conda deactivate
git clone https://github.com/Krecharles/tokengt_st
cd tokengt_st
conda env create -f environment.yml
conda activate tokengt_gpu
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cd ..
git clone https://github.com/krecharles/pytorch_geometric.git
cd pytorch_geometric/
git checkout add-token-gt
cd ..
pip install -e pytorch_geometric

cd tokengt_st
pip install wandb
wandb login

pip install rdkit-pypi==2022.9.5
pip install "numpy<2.0"

touch ~/.no_auto_tmux
git clone https://github.com/Krecharles/tokengt_st
git clone https://github.com/krecharles/pytorch_geometric.git
cd pytorch_geometric/
git checkout add-token-gt
cd ..
pip install -e pytorch_geometric

git clone https://github.com/Krecharles/tokengt_st
cd tokengt_st
git clone https://github.com/krecharles/pytorch_geometric.git
cd pytorch_geometric/
git checkout add-token-gt
cd ..
pip install -e pytorch_geometric
pip install numpy pytorch-lightning tqdm wandb networkx matplotlib einops rdkit==2023.9.1 ogb
python3 -m wandb login
python3 -m tokengt_paper_experiments.tgtp_exp
