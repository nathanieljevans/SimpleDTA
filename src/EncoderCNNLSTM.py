'''
This should work for both drug and protein
'''

import torch
from torch import nn


class EncoderCNNLSTM(torch.nn.Module):
    def __init__(self, embedding_dim, num_embeddings, padding_idx, kernel_size, conv_channels, latent_channels, hidden_channels, dropout=0., embedding_type='trainable'):
        super().__init__()

        self.embedding_type = embedding_type
        self.num_embeddings = num_embeddings

        if self.embedding_type == 'trainable': 
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        else: 
            embedding_dim = num_embeddings

        self.fs = nn.ModuleList([torch.nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=conv_channels, kernel_size=ksize, padding='same', bias=False),  
                                                        nn.ReLU(), 
                                                        nn.Dropout(dropout),
                                                        nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels*2, kernel_size=ksize, padding='same', bias=False),  
                                                        nn.ReLU(), 
                                                        nn.Dropout(dropout),
                                                        nn.Conv1d(in_channels=conv_channels*2, out_channels=conv_channels*3, kernel_size=ksize, padding='same', bias=False),  
                                                        nn.ReLU(),
                                                        nn.Dropout(dropout)) for ksize in kernel_size])

        self.lstm = nn.LSTM(input_size=conv_channels*3*len(kernel_size), hidden_size=hidden_channels, num_layers=1, bias=False, batch_first=True, dropout=dropout)

        
        self.lin = nn.Linear(hidden_channels, latent_channels)
        self.do = nn.Dropout(dropout)

    def _embed(self, x): 
        if self.embedding_type == 'trainable': 
            x = self.embedding(x)# shape (B, max_aa_len, embed dim)
        elif self.embedding_type == 'onehot': 
            x = torch.tensor(nn.functional.one_hot(x, num_classes=self.num_embeddings), dtype=torch.float)
        else: 
            raise Exception('Embedding type not recognized')

        return self.do(torch.permute(x, (0, 2, 1)))# conv expects: (N, Cin, Lin) 

    def forward(self, x): 
        ''''''
        z = self._embed(x)
        z = torch.cat([f(z) for f in self.fs], dim=1) # (B, C, L)
        z = torch.permute(z, (0,2,1))
        _, (h, _) = self.lstm(z)
        return self.lin(h[-1, :, :]) 