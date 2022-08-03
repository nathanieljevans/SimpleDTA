import torch 
from EncoderCNN import EncoderCNN
from EncoderLSTM import EncoderLSTM
from EncoderCNNLSTM import EncoderCNNLSTM
from NN import NN 
import config 


class SimpleDTA(torch.nn.Module):
    def __init__(self, embedding_dim, drug_kernel_size, prot_kernel_size, hidden_channels, latent_channels, encoder='lstm', mlp_out_channels=500, dropout=0., embedding_type='trainable'): 
        ''''''
        super().__init__()

        if encoder == 'cnn': 
            self.drug_encoder = EncoderCNN(embedding_dim=embedding_dim, 
                                        num_embeddings=len(config.smiles_options)+1, 
                                        padding_idx=len(config.smiles_options), 
                                        kernel_size=drug_kernel_size, 
                                        conv_channels=hidden_channels, 
                                        latent_channels=latent_channels,
                                        dropout=dropout,
                                        embedding_type=embedding_type)

            self.prot_encoder = EncoderCNN(embedding_dim=embedding_dim, 
                                        num_embeddings=len(config.aa_options)+1, 
                                        padding_idx=len(config.aa_options), 
                                        kernel_size=prot_kernel_size, 
                                        conv_channels=hidden_channels, 
                                        latent_channels=latent_channels,
                                        dropout=dropout,
                                        embedding_type=embedding_type)
        elif encoder == 'lstm': 

            self.drug_encoder = EncoderLSTM(embedding_dim=embedding_dim, 
                                        num_embeddings=len(config.smiles_options)+1, 
                                        padding_idx=len(config.smiles_options), 
                                        hidden_channels = hidden_channels, 
                                        latent_channels=latent_channels,
                                        dropout=dropout,
                                        embedding_type=embedding_type)

            self.prot_encoder = EncoderLSTM(embedding_dim=embedding_dim, 
                                        num_embeddings=len(config.smiles_options)+1, 
                                        hidden_channels = hidden_channels, 
                                        padding_idx=len(config.smiles_options), 
                                        latent_channels=latent_channels,
                                        dropout=dropout,
                                        embedding_type=embedding_type)

        elif encoder == 'cnn-lstm': 

            self.drug_encoder = EncoderCNNLSTM(embedding_dim=embedding_dim, 
                                        num_embeddings=len(config.smiles_options)+1, 
                                        padding_idx=len(config.smiles_options), 
                                        kernel_size=drug_kernel_size, 
                                        conv_channels=hidden_channels, 
                                        latent_channels=latent_channels,
                                        hidden_channels=hidden_channels*4,
                                        dropout=dropout,
                                        embedding_type=embedding_type)
            
            self.prot_encoder = EncoderCNNLSTM(embedding_dim=embedding_dim, 
                                        num_embeddings=len(config.smiles_options)+1, 
                                        padding_idx=len(config.smiles_options), 
                                        kernel_size=drug_kernel_size, 
                                        conv_channels=hidden_channels, 
                                        latent_channels=latent_channels,
                                        hidden_channels=hidden_channels*4,
                                        dropout=dropout,
                                        embedding_type=embedding_type)

        else: 
            raise Exception(f'unrecognized encoder type: {encoder}')
            
        self.mlp_out =  NN(in_channels=(2*latent_channels), out_channels=1, num_layers=2, hidden_channels=mlp_out_channels, norm=True, dropout=dropout, bias=True)

    def forward(self, smiles_idx, aa_idx): 

        dz = self.drug_encoder(smiles_idx).tanh()
        pz = self.prot_encoder(aa_idx).tanh()
        z = torch.cat((dz, pz), dim=1)
        return self.mlp_out(z)