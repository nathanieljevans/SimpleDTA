'''
This should work for both drug and protein
'''

import torch
from torch import nn


class EncoderCNN(torch.nn.Module):
    def __init__(self, embedding_dim, num_embeddings, padding_idx, kernel_size, conv_channels, latent_channels, dropout=0., embedding_type='trainable'):
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

        self.pool = nn.AdaptiveMaxPool1d(output_size=1, return_indices=False)
        self.lin = nn.Linear(conv_channels*3*len(kernel_size), latent_channels)
        self.do = nn.Dropout(dropout)

        #self.lin = nn.LazyLinear(latent_channels)

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
        z1 = torch.cat([f(z) for f in self.fs], dim=1)#.view(x.size(0), -1) # shape (N, conv_channels*3, Lin)
        z2 = self.pool(z1).squeeze(-1)
        #x, _ = torch.max(x, dim=2)#.values # shape (N, conv_channels*3)
        return self.lin(z2) 

        
