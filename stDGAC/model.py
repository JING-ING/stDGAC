import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .layer import layer
from torch.nn import Linear

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand(
            (vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)

class GAC(torch.nn.Module):
    def __init__(self, input_dim, z_dim, graph_neigh, layerType='GATConv', heads=1, dropout=0.2, act=F.relu):
        super(GAC, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.disc = Discriminator(self.z_dim)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

        #### Encoder ####
        self.encode_conv1 = layer(layerType, in_channels=input_dim, out_channels=z_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.encode_bn1 = torch.nn.BatchNorm1d(z_dim)
        self.decode_conv4 = layer(layerType, in_channels=z_dim, out_channels=input_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.decode_bn2 = torch.nn.BatchNorm1d(input_dim)
        self.decode_linear2 = torch.nn.Linear(z_dim, input_dim)


    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = self.encode_bn1(self.encode_conv1(z, adj))
        hiden_emb = z
        emb = self.act(hiden_emb)

        h = self.decode_bn2(self.decode_conv4(hiden_emb, adj))

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = self.encode_bn1(self.encode_conv1(z_a, adj))
        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hiden_emb, h, ret, ret_a

class DAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, dropout=0):
        super(DAE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.encode_bn1 = torch.nn.BatchNorm1d(n_enc_1)
        self.encode_bn2 = torch.nn.BatchNorm1d(n_enc_2)
        #
        self.z_layer = Linear(n_enc_2, n_z)
        #
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        # bn
        self.decode_bn1 = torch.nn.BatchNorm1d(n_dec_1)
        self.decode_bn2 = torch.nn.BatchNorm1d(n_dec_2)

        self.x_bar_layer = Linear(n_dec_2, n_input)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))

        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)
        return x_bar, enc_h1, enc_h2, z

    def loss_function(self, x_bar, x):
        reconstruction_loss = nn.MSELoss()(x_bar, x)
        total_loss = reconstruction_loss

        return total_loss