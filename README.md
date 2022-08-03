# SimpleDTA

> Prediction Task: Take drug SMILES and protein amino acid sequences as input and predict the binding affintiy. 

This will then be used to predict binding affinities between the proteins and drugs specified by `/extdata/mytargets.csv` and `/extdata/mydrugs.csv`. The predicted results will be combined with the CLUE Compound repurposing dataset. It is recommended that you predict on `warm proteins`, e.g., proteins that are included in the training set - as prediction performance on out-of-training proteins is poor. To limit prediction on warm proteins, use `--dataset XXX` in `predict.py`; to predict on all of `mytargets`, pass `--dataset none`. 

# Quickstart 

See `src/config.py` for relevant parameters. 

```bash 
conda env create --file environment.yml 

conda activate SimpleDTA 

cd scripts 

python train.py --embedding_dim 50 --drug_kernel_size 3 5 7 --prot_kernel_size 3 7 15 --hidden_channels 32 --latent_channels 124 --mlp_channels 512 --lr 1e-3 --dropout 0.1 --embedding_type trainable --encoder cnn --embedding_type trainable --out ../output/models/ --dataset BindingDB_IC50 --exp_id cnn_001 --batch_size 256 --workers 2 --epochs 50 --split cold_drug

python predict.py --model ../output/models/cnn_001/model.pt --targets ../extdata/mytargets.csv --drugs ../extdata/mydrugs.csv --out ../output/cnn_results_test.csv --batch_size 100 --num_workers 2 --dataset BindingDB_IC50

# TODO: python agg
```

The results will be filtered to Drug-Target interactions with a value greater than `affinity_threshold`, specified in `config.py` (default=1000; units in nm). 

Results: 

| gene_symbol | pert_id | pIC50 | IC50 | cmap_name | moa | in_clue |   
|-|-|-|-|-|-|-|
ABL2 | BRD-A30655177 | 4.968025   
... | ... | ... |  
ABL2 | BRD-A67373739 | 4.9967165   


> `gene_symbol`:   
> `pert_id`:   
> `pIC50`:   
> `IC50`: 
> `cmap_name`:   
> `moa`:  
> `in_clue`:  


# Drug + Protein Encoding 

Drug SMILES and protein amino acid sequences will be encoded by (optionally) one-hot or word embeddings, and then (optionally) independantly fed through a cnn, lstm or cnn->lstm before being concatenated for the output layer. 

# Output layer 

The Drug encoding and Protein encoding will be concatenated and a fully connected layer will predict binding affinity as a regression problem. The endogenous variable (kd) will be transformed to for prediction by: 

$$ y = pKd = -log10(kd) $$

# Training Data 

We will use the data as made available by the [therapeutic data commons](https://tdcommons.ai/multi_pred_tasks/dti/). This contains three datasets (info taken from `tdcommons`): 

NOTE: Each dataset has a different prediction metric, we will combine all 3 datasets using a multioutput prediction. 

(information below copied from `tdcommons`)

## BindingDB 

> Dataset Description: BindingDB is a public, web-accessible database of measured binding affinities, focusing chiefly on the interactions of protein considered to be drug-targets with small, drug-like molecules.

> Task Description: Regression. Given the target amino acid sequence/compound SMILES string, predict their binding affinity.

> Dataset Statistics: (# of DTI pairs, # of drugs, # of proteins) 52,284/10,665/1,413 for Kd, 991,486/549,205/5,078 for IC50, 375,032/174,662/3,070 for Ki.

## DAVIS

> Dataset Description: The interaction of 72 kinase inhibitors with 442 kinases covering >80% of the human catalytic protein kinome.

> Task Description: Regression. Given the target amino acid sequence/compound SMILES string, predict their binding affinity.

> Dataset Statistics: 0.3.2 Update: 25,772 DTI pairs, 68 drugs, 379 proteins. Before: 27,621 DTI pairs, 68 drugs, 379 proteins.

## KIBA

> Dataset Description: Toward making use of the complementary information captured by the various bioactivity types, including IC50, K(i), and K(d), Tang et al. introduces a model-based integration approach, termed KIBA to generate an integrated drug-target bioactivity matrix.

> Task Description: Regression. Given the target amino acid sequence/compound SMILES string, predict their binding affinity.

> Dataset Statistics: 0.3.2 Update: 117,657 DTI pairs, 2,068 drugs, 229 proteins. Before: 118,036 DTI pairs, 2,068 drugs, 229 proteins.

# References 

Tang J, Szwajda A, Shakyawar S, et al. Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. J Chem Inf Model. 2014;54(3):735-743.

Huang, Kexin, et al. “DeepPurpose: a Deep Learning Library for Drug-Target Interaction Prediction” Bioinformatics.

Davis, M., Hunt, J., Herrgard, S. et al. Comprehensive analysis of kinase inhibitor selectivity. Nat Biotechnol 29, 1046–1051 (2011).

Liu, Tiqing, et al. “BindingDB: a web-accessible database of experimentally determined protein–ligand binding affinities.” Nucleic acids research 35.suppl_1 (2007): D198-D201.