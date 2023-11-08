from math import cos, pi
import torch
from .preprocess import preprocess_adj, preprocess, construct_interaction, add_contrastive_label, get_feature, disturb, fix_seed
import numpy as np
from .model import GAC, DAE
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import Adam

class MyDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx].float(), torch.from_numpy(np.array(idx))

def add_noise(inputs):
    noise_value = 0.9
    return inputs + (torch.randn(inputs.shape).cuda() * noise_value)

def pretrain_dae(model,
                feature,
                weight_decay=0,
                EPOCH=400,  # 400
                ):
    dataset = MyDataset(feature)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)
    train_loss = []

    for epoch in tqdm(range(EPOCH)):
        if epoch + 1 < EPOCH:
            for batch_idx, (x, _) in enumerate(train_loader):
                x_noise = add_noise(x)
                x_noise = x_noise.cuda()
                x = x.cuda()

                x_bar, _, _, z = model(x_noise)
                loss = F.mse_loss(x_bar, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            new_features = []
            for batch_idx, (x, _) in enumerate(train_loader):
                x_noise = add_noise(x)
                x_noise = x_noise.cuda()
                x = x.cuda()

                x_bar, _, _, z = model(x_noise)
                new_features.append(z.detach())
                loss = F.mse_loss(x_bar, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
    features = new_features[0]
    for i in range(1, len(new_features)):
        features = torch.cat((features, new_features[i]), dim=0)
    features = features.detach()
    assert features.size(0) == feature.size(0)

    return features

class stDGAC():
    def __init__(self, adata, random_seed=50, device='cuda:0'):
       self.adata = adata.copy()
       self.random_seed = random_seed
       self.add_regularization = False
       self.device = device
       
       fix_seed(self.random_seed)
       preprocess(self.adata)
       construct_interaction(self.adata)
       add_contrastive_label(self.adata)
       self.adata_output = self.adata.copy()
    
    def train_stDGAC(self):
        fix_seed(self.random_seed)
        adata = self.adata_output.copy()
        get_feature(adata)
        print('Begin to train...')
        model = Train(adata, device=self.device)
        emb = model.train()
        self.adata_output.obsm['emb'] = emb

        return self.adata_output
    
class Train():
    def __init__(self,
                 adata,
                 device='cuda:0',
                 learning_rate=0.001,
                 weight_decay=0,
                 epochs=600,
                 dim_d1=1024,
                 dim_d2=512,
                 dae_output=256,
                 gac_output=64,
                 alpha=10,
                 beta=1,
                 ):

        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        
        self.features = torch.FloatTensor(adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(self.device)
        self.adj = adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        self.dim_d1 = dim_d1
        self.dim_d2 = dim_d2
        self.dae_output = dae_output
        self.dae_input = self.features.shape[1]
        self.gac_output = gac_output

        self.adj = preprocess_adj(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)
            
    def train(self):
        self.model_DAE = DAE(self.dim_d1, self.dim_d2, self.dim_d2, self.dim_d1, self.dae_input, self.dae_output).to(self.device)
        self.model_DAE.train()
        self.features_dae = pretrain_dae(model=self.model_DAE, feature=self.features).to(self.device)
        self.gac_input = self.dae_output
        self.model_GAC = GAC(self.gac_input, self.gac_output, self.graph_neigh).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model_GAC.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)
        self.model_GAC.train()
        for epoch in tqdm(range(self.epochs)): 
            self.model_GAC.train()
            self.features_a = disturb(self.features_dae)
            self.hiden_feat, self.emb, ret, ret_a = self.model_GAC.forward(self.features_dae, self.features_a, self.adj)

            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features_dae, self.emb)
            loss = self.alpha * self.loss_feat + self.beta * (self.loss_sl_1 + self.loss_sl_2)
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        print('Training finished!')
        
        with torch.no_grad():
             self.model_DAE.eval()
             self.model_GAC.eval()
             x = self.features.cuda()
             x_bar, _, _, z = self.model_DAE(x)
             self.emb_rec = self.model_GAC(z, self.features_a, self.adj)[1].detach().cpu().numpy()

             return self.emb_rec

