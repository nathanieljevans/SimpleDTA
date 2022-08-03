'''

python predict.py --model ../output/models/cnn_001/model.pt --targets ../extdata/mytargets.csv --drugs ../extdata/mydrugs.csv --out ../output/cnn_results_test.csv --batch_size 100 --num_workers 2 --dataset BindingDB_IC50
'''

import argparse
import os
import torch
import numpy as np
from SimpleDTA import SimpleDTA
import torch
from matplotlib import pyplot as plt 
import torch
import os
import pandas as pd 
from DTIDataset import DTIDataset
import config 
import torch_geometric
from tdc.multi_pred import DTI


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str,
                        help="path to model")

    parser.add_argument("--targets", type=str,
                        help="path to targets csv")

    parser.add_argument("--drugs", type=str,
                        help="path to drugs csv")

    parser.add_argument("--out", type=str,
                        help="path to save results")

    parser.add_argument("--batch_size", type=int,
                        help="batch size of dataloader")

    parser.add_argument("--num_workers", type=int,
                        help="data loader # workers")

    parser.add_argument("--dataset", type=str,
                        help="dataset use to train model - will filter `targets` to those included in the training data. Use none to predict on all targets.")

    return parser.parse_args()

if __name__ == '__main__': 
    
    args = get_args()

    ########################################### DEVICE #####################################################

    if torch.cuda.is_available():
        device = 'cuda:0'
    else: 
        device = 'cpu'

    print('predicting on device:', device)

    ############################################ DATA LOADERS #####################################################

    mytargets = pd.read_csv(args.targets)
    if args.dataset == 'none':
        pass
    else: 
        data = DTI(name = args.dataset, path='../data/')
        training_targets = data.harmonize_affinities(mode = 'max_affinity').Target_ID.unique()
        mytargets = mytargets[lambda x: x.uniprot_id.isin(training_targets)]
        print('# warm targets (e.g., proteins present in training data):', len(set(training_targets).intersection(set(mytargets.uniprot_id))))

    mydrugs = pd.read_csv(args.drugs)
    split = mytargets[['gene_symbol','uniprot_id', 'Sequence']].merge(mydrugs[['pert_id', 'canonical_smiles']], how='cross')
    split = split.rename({'Sequence':'Target', 'canonical_smiles': 'Drug'}, axis=1)

    aa_options = config.aa_options
    aa_index = {**{aa:aa_options.index(aa) for aa in aa_options}, **{config.padding_char:len(config.aa_options)}}
    smiles_options = config.smiles_options
    smiles_index = {**{ch:smiles_options.index(ch) for ch in smiles_options}, **{config.padding_char:len(config.smiles_options)}}

    dataset = DTIDataset(split=split, aa_index=aa_index, smiles_index=smiles_index, max_aa_len=config.max_aa_len, max_smiles_len=config.max_smiles_len, padding_char=config.padding_char, mode='prediction' )
    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    print('# DTI candidates:', len(split))

    ############################################ MODEL INIT #####################################################

    model = torch.load(args.model).eval().to(device)

    ############################################ PREDICT #####################################################

    _yhat = []
    _drug = []
    _target= []
    for i,data in enumerate(dataloader): 
        print(f'progress: {i}/{len(dataloader)} \t [{100*i/len(dataloader):.0f}%]', end='\r')
        data.to(device)
        _yhat.append( model(data.smiles_idx, data.aa_idx).squeeze(-1).detach().cpu().numpy() )
        _drug.append(np.array(data.drug))
        _target.append(np.array(data.target))
    print()
    split = split.assign(pred_pIC50 = np.concatenate(_yhat), drug_id_check=np.concatenate(_drug), target_id_check=np.concatenate(_target))
    
    assert (split.uniprot_id.values == split.target_id_check.values).all(), 'target id check failed'
    assert (split.pert_id.values == split.drug_id_check).all(), 'drug id check failed'
    
    split = split[['gene_symbol', 'uniprot_id', 'pert_id', 'pred_pIC50']]
    
    #split = split.groupby(['gene_symbol', 'pert_id']).agg(['mean', 'min', 'max', 'count']).reset_index()
    #split = split[lambda x: x[('pred_IC50', 'min')] <= config.affinty_threshold]

    print('saving to:', args.out)
    split.to_csv(args.out, index=False)


    
