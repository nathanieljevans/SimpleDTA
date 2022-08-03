'''
This should work for both drug and protein
'''

import torch
from torch import nn


class EncoderLSTM(torch.nn.Module):
    def __init__(self, embedding_dim, num_embeddings, padding_idx, hidden_channels, latent_channels, dropout=0., embedding_type='trainable'):
        super().__init__()

        self.embedding_type = embedding_type
        self.num_embeddings = num_embeddings

        if self.embedding_type == 'trainable': 
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        else: 
            embedding_dim = num_embeddings

        self.f = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_channels, num_layers=1, bias=False, batch_first=True, dropout=dropout)
        self.do = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels, latent_channels)

    def _embed(self, x): 
        if self.embedding_type == 'trainable': 
            x = self.embedding(x)# shape (B, max_aa_len, embed dim)
        elif self.embedding_type == 'onehot': 
            x = torch.tensor(nn.functional.one_hot(x, num_classes=self.num_embeddings), dtype=torch.float)
        else: 
            raise Exception('Embedding type not recognized')

        return self.do(x) #torch.permute(x, (0, 2, 1)) # lstm expects: (batch, seq, feature)

    def forward(self, x): 
        ''''''
        z = self._embed(x)
        _, (h, _) = self.f(z)
        return self.lin(h[-1, :, :]) 

        
