import torch 


class NN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels, norm=True, dropout=0., bias=True): 
        ''''''

        super().__init__()

        seq = []

        act = torch.nn.ELU

        # first layer 
        seq.append(torch.nn.Linear(in_channels, hidden_channels, bias=bias))
        seq.append(torch.nn.Dropout(dropout))
        seq.append(act())

        for l in range(num_layers - 1): 
            if norm: seq.append(torch.nn.BatchNorm1d(int(hidden_channels / (l+1))))
            seq.append(torch.nn.Linear(int(hidden_channels / (l+1)), int(hidden_channels / (l+2)), bias=bias))
            seq.append(torch.nn.Dropout(dropout))
            seq.append(act())
            
        # output layer
        if norm: seq.append(torch.nn.BatchNorm1d(int(hidden_channels / (l+2))))
        seq.append(torch.nn.Linear(int(hidden_channels / (l+2)), out_channels, bias=bias))

        self.f = torch.nn.Sequential(*seq)

    def forward(self, x): 

        return self.f(x)