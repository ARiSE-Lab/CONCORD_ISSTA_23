# CONCORD_ISSTA23
Code for ISSTA'23 paper "CONCORD: Clone-aware Contrastive Learning for Source Code"

## Environment Setup
```bash
conda create -n concord Python=3.8.12;
conda activate concord;

# install torch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge;

# install apex
git clone https://github.com/NVIDIA/apex.git;
cd apex/;
git checkout feae3851a5449e092202a1c692d01e0124f977e4;
pip install -v --disable-pip-version-check --no-cache-dir ./;
cd ../

# install pip packages
cd CONCORD_ISSTA23;
pip install -r requirements.txt;
export PYTHONPATH=$(pwd);
```

## Model Weights and Data

You could find the pre-trained weights of the main CONCORD model and task-specific data here:

- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8393793.svg)](https://doi.org/10.5281/zenodo.8393793)

__Important Note For Data Pre-processing:__ During CONCORD's pre-training, we need to align the syntax labels with code tokens, which requires the data pre-processing. To avoid the distribution shift, the task-specific fine-tuning data needs to do the same pre-processing with the following two steps:
- Parse the source code with [Tree-sitter](https://github.com/tree-sitter/py-tree-sitter) and tokenize the sequence following the grammar of corresponding programming languages. 
- Sub-tokenize with the pre-trained [BPE model](vocab/multilingual_5k_repo_50k_vocab.model).

Check out our data samples (finetune_data.zip) for the expected format of the pre-processed code.
