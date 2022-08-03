from dataclasses import dataclass
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import pickle as pkl
import time as tm
import pandas as pd
from from_smiles import from_smiles

class DTIDataset(Dataset):
    def __init__(self, split, aa_index, smiles_index, max_aa_len, max_smiles_len, padding_char, mode='training'):
        '''
        '''
        super().__init__()
        assert mode in ['training', 'prediction'], 'mode should be one of "training" or "prediction"'
        self.mode = mode
        self.aa_index = aa_index
        self.smiles_index = smiles_index
        self.split = split 
        self.max_aa_len = max_aa_len
        self.padding_char = padding_char
        self.max_smiles_len = max_smiles_len

    def len(self):
        return self.split.shape[0]

    def _get_prediction_data(self, idx):
        '''this is intended for prediction of DTA on `mydrugs` and `mytargets`, no label'''
        obs = self.split.iloc[idx]
        data = Data() 
        data.drug = obs.pert_id
        data.target = obs.uniprot_id
        data.smiles_idx = self._smiles2idx(obs.Drug)
        data.aa_idx = self._aa2idx(obs.Target)
        return data 

    def _get_training_data(self, idx): 
        obs = self.split.iloc[idx]
        data = Data() 
        data.smiles = obs.Drug
        data.smiles_idx = self._smiles2idx(obs.Drug)
        data.aa_idx = self._aa2idx(obs.Target)
        data.y = torch.tensor(obs.Y, dtype=torch.float) 
        if torch.isnan(data.y).any(): raise Exception('y is nan')
        if torch.isinf(data.y).any(): raise Exception('y is inf')
        return data

    def get(self, idx):
        ''''''
        if self.mode == 'training': 
            data = self._get_training_data(idx)
        elif self.mode == 'prediction': 
            data = self._get_prediction_data(idx)
        else: 
            raise Exception ('unrecognized mode')
        return data

    def _smiles2idx(self, smiles): 

        if len(smiles) > self.max_aa_len: 
            smiles = smiles[0:self.max_smiles_len]
        else: 
            smiles = (self.padding_char*(self.max_aa_len - len(smiles))) + smiles

        smiles = [self.smiles_index[char] for char in smiles]

        return torch.tensor(smiles, dtype=torch.long).unsqueeze(0)

    def _aa2idx(self, aas): 
        
        if len(aas) > self.max_aa_len: 
            # cut down if too long
            aas = aas[0:self.max_aa_len]
        else: 
            # pad beginning
            aas = (self.padding_char*(self.max_aa_len - len(aas))) + aas 

        aas = [self.aa_index[aa] for aa in aas]

        return torch.tensor(aas, dtype=torch.long).unsqueeze(0)