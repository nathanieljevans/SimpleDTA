

import torch
from torch_geometric.nn import GCNConv,  global_max_pool, GATv2Conv, PairNorm



class DrugEncoderGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, num_layers, conv, edge_dim, pairnorm=True, dropout=0.):
        super().__init__()

        self.pairnorm = pairnorm
        self.norm = PairNorm()
        self.conv = conv 
        self.do = torch.nn.Dropout(dropout)
        self.latent_channels = latent_channels

        if conv == 'GCN': 
            self.convs = torch.nn.ModuleList([GCNConv(in_channels, hidden_channels, cached=False)] + [GCNConv(hidden_channels, hidden_channels, cached=False) for x in range(num_layers - 1)])
        elif conv == 'GAT': 
            self.convs = torch.nn.ModuleList([GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)] + [GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim) for x in range(num_layers - 2)] + [GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim)])

        self.nn = torch.nn.Sequential(torch.nn.BatchNorm1d(hidden_channels), 
                                      torch.nn.Linear(hidden_channels, latent_channels))

    def forward(self, x, edge_index, edge_attr, batch): 

        for i,conv in enumerate(self.convs): 

            if (i != 0) & self.pairnorm: 
                x = self.norm(x)

            if self.conv == 'GCN': 
                x = conv(x, edge_index).relu()
            else: 
                x = conv(x, edge_index, edge_attr).relu()

            x = self.do(x)

        x = global_max_pool(x, batch)


        return self.nn(x)
