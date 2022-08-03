
from tdc.multi_pred import DTI
from DTIDataset import DTIDataset
import torch_geometric 
import config
import numpy as np


def get_dataloaders(batch_size, num_workers, dataset='BindingDB_Kd', split='cold_drug', seed=np.random.randint(1000)): 
    
    print('training on dataset:', dataset)
    data = DTI(name = dataset, path='../data/')
    if dataset in ['BindingDB_Kd', 'BindingDB_IC50']: 
        data.convert_to_log(form = 'binding')

    if split == 'cold_drug': 
        split = data.get_split(method='cold_split', column_name=['Drug'], seed=seed)
    if split == 'cold_drug_target':
        split = data.get_split(method='cold_split', column_name=['Drug', 'Target'], seed=seed)
    elif split == 'random': 
        split = data.get_split(method='random')
    else:
        raise Exception('split should be one of: "cold_drug" or "random"')

    aa_options = config.aa_options
    aa_index = {**{aa:aa_options.index(aa) for aa in aa_options}, **{config.padding_char:len(config.aa_options)}}

    smiles_options = config.smiles_options
    smiles_index = {**{ch:smiles_options.index(ch) for ch in smiles_options}, **{config.padding_char:len(config.smiles_options)}}

    train_dataset = DTIDataset(split=split['train'], aa_index=aa_index, max_aa_len=config.max_aa_len, padding_char=config.padding_char, max_smiles_len=config.max_smiles_len, smiles_index=smiles_index)
    test_dataset = DTIDataset(split=split['test'], aa_index=aa_index, max_aa_len=config.max_aa_len, padding_char=config.padding_char, max_smiles_len=config.max_smiles_len, smiles_index=smiles_index)
    valid_dataset = DTIDataset(split=split['valid'], aa_index=aa_index, max_aa_len=config.max_aa_len, padding_char=config.padding_char, max_smiles_len=config.max_smiles_len, smiles_index=smiles_index)

    print()
    print('train dataset size:', len(train_dataset))
    print('test dataset size:', len(test_dataset))
    print('valid dataset size:', len(valid_dataset))
    print('number of unique DRUGS in training set:   \t', split['train'].Drug.unique().shape[0])
    print('number of unique PROTEINS in training set:\t', split['train'].Target.unique().shape[0])

    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch_geometric.loader.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_loader, test_loader, valid_loader