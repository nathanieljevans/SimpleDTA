'''

python train.py --embedding_dim 50 --drug_kernel_size 3 5 7 --prot_kernel_size 3 7 15 --hidden_channels 32 --latent_channels 124 --mlp_channels 512 --lr 1e-3 --dropout 0.1 --embedding_type trainable --encoder cnn --embedding_type trainable --out ../output/models/ --dataset BindingDB_IC50 --exp_id cnn_001 --batch_size 256 --workers 2 --epochs 50 --split cold_drug

'''
import argparse
import os
import shutil
import torch
from tdc.multi_pred import DTI
import numpy as np
from get_dataloaders import get_dataloaders
from SimpleDTA import SimpleDTA
import torch
from matplotlib import pyplot as plt 
from sklearn.metrics import r2_score
import time 

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os


class MyWriter(): 
    def __init__(self, exp, root): 
        ''''''
        if not os.path.exists(root): os.mkdir(root)

        self.exp    = exp
        self.writer = SummaryWriter(log_dir = root, comment=exp)

    def log(self, epoch, train_loss, train_r2, train_corr, test_loss, test_r2, test_corr):

        self.writer.add_scalars(main_tag        = 'train-loss',
                               tag_scalar_dict  = {self.exp:train_loss}, 
                               global_step      = epoch)

        self.writer.add_scalars(main_tag        = 'test-loss',
                               tag_scalar_dict  = {self.exp:test_loss}, 
                               global_step      = epoch)

        self.writer.add_scalars(main_tag        = 'train-r2',
                               tag_scalar_dict  = {self.exp:train_r2}, 
                               global_step      = epoch)

        self.writer.add_scalars(main_tag        = 'test-r2',
                               tag_scalar_dict  = {self.exp:test_r2}, 
                               global_step      = epoch)

        self.writer.add_scalars(main_tag        = 'train-corr',
                               tag_scalar_dict  = {self.exp:train_corr}, 
                               global_step      = epoch)

        self.writer.add_scalars(main_tag        = 'test-corr',
                               tag_scalar_dict  = {self.exp:test_corr}, 
                               global_step      = epoch)

def myprint(s, padding=12): 
    s = str(s)

    if len(s) % 2 != 0: 
        s += ' '

    sz = int(len(s)/2)
    pad = ' '*(padding-sz)
    return '|' + pad + str(s) + pad + '|'


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_dim", type=int,
                        help="number of channels for the embedding")

    parser.add_argument("--drug_kernel_size", type=int, nargs='+',
                        help="kernel size(s), ex: 4 9 13")

    parser.add_argument("--prot_kernel_size", type=int, nargs='+',
                        help="kernel size(s), ex: 4 9 13")

    parser.add_argument("--hidden_channels", type=int,
                        help="numer of channels to use for hidden layers")

    parser.add_argument("--latent_channels", type=int,
                        help="numer of channels to use for hidden layers")

    parser.add_argument("--mlp_channels", type=int,
                        help="numer of channels to use for mlp out")

    parser.add_argument("--lr", type=float,
                        help="learning rate")

    parser.add_argument("--dropout", type=float,
                        help="dropout [0,1]")

    parser.add_argument("--encoder", type=str,
                        help="type of encoder to use, options: 'cnn', 'lstm', 'cnn-lstm'")

    parser.add_argument("--embedding_type", type=str,
                        help="can be 'onehot' or 'trainable'")

    parser.add_argument("--out", type=str,
                        help="output directory; where to save model and results")

    parser.add_argument("--dataset", type=str,
                        help="which dataset to use, options: 'BindingDB_Kd', 'KIBA', 'DAVIS' ")

    parser.add_argument("--split", type=str,
                        help="how to split the training data; cold splits separate drugs/proteins into train/test/val. Options: 'random', 'cold_drug', 'cold_drug_protein'")

    parser.add_argument("--exp_id", type=str,
                        help="experiment identifier")

    parser.add_argument("--batch_size", type=int,
                        help="number of observations per batch")

    parser.add_argument("--workers", type=int,
                        help="number of cpu workers per data loader")

    parser.add_argument("--epochs", type=int,
                        help="number of training epochs")

    return parser.parse_args()


if __name__ == '__main__': 
    
    args = get_args()

    if not os.path.exists(args.out): 
        os.mkdir(args.out)

    save_dir = args.out + '/' + args.exp_id
    if os.path.exists(save_dir): 
        print('experiment dir exists, deleting contents...')
        shutil.rmtree(save_dir)

    os.mkdir(save_dir)

    print()
    print(args)
    with open(save_dir + '/args.txt', 'w') as f: 
        f.write(str(args))

    ########################################### DEVICE #####################################################

    if torch.cuda.is_available():
        device = 'cuda:0'
    else: 
        device = 'cpu'

    print()
    print('training on device:', device)

    ############################################ DATA LOADERS #####################################################

    train_loader, test_loader, val_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.workers, dataset=args.dataset, split=args.split)

    ############################################ MODEL INIT #####################################################


    model = SimpleDTA(embedding_dim     =   args.embedding_dim, 
                        drug_kernel_size  =   args.drug_kernel_size, 
                        prot_kernel_size  =   args.prot_kernel_size, 
                        hidden_channels   =   args.hidden_channels, 
                        latent_channels   =   args.latent_channels, 
                        mlp_out_channels  =   args.mlp_channels, 
                        dropout           =   args.dropout, 
                        encoder           =   args.encoder, 
                        embedding_type    =   args.embedding_type).to(device)

    ### 

    print()
    print('drug enc:', sum(p.numel() for p in model.drug_encoder.parameters()))
    print('prot enc:', sum(p.numel() for p in model.prot_encoder.parameters()))
    print('mlp out:', sum(p.numel() for p in model.mlp_out.parameters()))
    
    ###

    optim               = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit                = torch.nn.MSELoss()
    sched               = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel',cooldown=5, min_lr=1e-6, verbose=True)

    writer              = MyWriter(exp=args.exp_id, root=save_dir)

    ########################################################################
    ########################################################################

    print()
    print('(train, test)')
    print(''.join([myprint(x) for x in ['EPOCH', 'LOSS', 'R2', 'CORR', 'TIME']]))

    for epoch in range(args.epochs): 
        
        loss_avg = 0 
        tic = time.time() 
        _y_train = []
        _yhat_train = []

        for data in train_loader: 
            optim.zero_grad()
            model.train()
            data.to(device)
            yhat = model(data.smiles_idx, data.aa_idx).squeeze(-1)
                
            loss = crit(yhat, data.y)
            loss.backward() 
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optim.step()

            loss_avg += loss.detach().cpu().numpy().item() / len(train_loader)
            _y_train.append( data.y.detach().cpu().numpy() )
            _yhat_train.append( yhat.detach().cpu().numpy() ) 

        elapsed = time.time() - tic 
        tic = time.time()

        with torch.no_grad(): 
            model.eval()
            test_loss = 0 
            _y_test = []
            _yhat_test = []
            for data in test_loader: 
                data.to(device)
                yhat = model(data.smiles_idx, data.aa_idx).squeeze(-1)
                test_loss += crit(yhat, data.y).detach().cpu().numpy().item() / len(test_loader)

                _y_test.append( data.y.detach().cpu().numpy() )
                _yhat_test.append( yhat.detach().cpu().numpy() ) 

        # calc metrics -- train 
        y_train = np.concatenate(_y_train)
        yhat_train = np.concatenate(_yhat_train)
        r2_train = r2_score(y_train, yhat_train)
        corr_train = np.corrcoef(y_train, yhat_train)[0,1]

        #              -- test
        y_test = np.concatenate(_y_test)
        yhat_test = np.concatenate(_yhat_test)
        r2_test = r2_score(y_test, yhat_test)
        corr_test = np.corrcoef(y_test, yhat_test)[0,1]
        # 
        elapsed2 = time.time() - tic
        tic = time.time()

        writer.log(epoch, loss_avg, r2_train, corr_train, test_loss, r2_test, corr_test)

        sched.step(test_loss)

        print(''.join([myprint(x) for x in [epoch, f'({loss:.4f}, {test_loss:.4f})', f'({r2_train:.4f}, {r2_test:.4f})', f'({corr_train:.4f}, {corr_test:.4f})', f'({elapsed:.1f} s, {elapsed2:.1f} s)']]))

        torch.save(model, save_dir + '/model.pt')


    ########################################################################
    ########################################################################
    print()
    print('training complete... evaluating on validation data')

    with torch.no_grad(): 
        model.eval()
        val_loss = 0 
        _y_val = []
        _yhat_val = []
        for data in val_loader: 
            data.to(device)
            yhat = model(data.smiles_idx, data.aa_idx).squeeze(-1)
            val_loss += crit(yhat, data.y).detach().cpu().numpy().item() / len(val_loader)

            _y_val.append( data.y.detach().cpu().numpy() )
            _yhat_val.append( yhat.detach().cpu().numpy() ) 

        # calc metrics -- train 
        y_val = np.concatenate(_y_val)
        yhat_val = np.concatenate(_yhat_val)
        r2_val = r2_score(y_val, yhat_val)
        corr_val = np.corrcoef(y_val, yhat_val)[0,1]

        print()
        print('VALIDATION DATASET METRICS')
        print(f'pearson corr: {corr_val:.3f}')
        print(f'R2: {r2_val:.3f}')
        print(f'MSE (batch avg): {val_loss:.4f}')

        with open(save_dir + '/valid_metrics.csv', 'w') as f: 
            f.write('pearson,r2,mse\n')
            f.write(f'{corr_val:.3f},{r2_val:.3f},{val_loss:.4f}')



