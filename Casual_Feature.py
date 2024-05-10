
import torch.nn as nn
import torch.nn.functional as F
from Point_loader import *
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
import warnings
from util import jaccard_similarity,pca_feature_with_mean_replacement

warnings.filterwarnings("ignore", category=UserWarning)


def asymmetric_l2_loss(u, tau, alpha=0.5):
    positive_errors = torch.relu(u)
    negative_errors = torch.relu(-u)
    loss = torch.mean(alpha * tau * positive_errors ** 2 + (1 - alpha) * (1 - tau) * negative_errors ** 2)
    return loss 

class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))

        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]

        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.n_head)
                + " -> "
                + str(self.f_in)
                + " -> "
                + str(self.f_out)
                + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32,eps=1e-4),
            torch.nn.InstanceNorm1d(64,eps=1e-4),
        ]

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x 

        bs, n = x.size()[:2]

        for i, gat_layer in enumerate(self.layer_stack):

            if bs > 1:
                x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=2)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)

        return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding):
        graph_embeded_data = self.gat_net(obs_traj_embedding)
        return graph_embeded_data


class CausalFeatureExtractor(nn.Module):
    def __init__(self, input_dim, gat_output_dim, mlp_hidden_dim, device, n_heads=2):
        super(CausalFeatureExtractor, self).__init__()
        self.gat_out_dim = gat_output_dim
        self.device = device
        self.gat_encoder = GATEncoder(
            n_units=[input_dim, gat_output_dim],
            n_heads=[n_heads, 1],
            dropout=0.1,
            alpha=0.2
        )

        self.Decoder = Decoder(
            input_dim=input_dim,
            hidden_dims=mlp_hidden_dim
        ).to(self.device)

        self.cross_attention = CrossAttention(
            input_dim=gat_output_dim,
            out_dim=input_dim,
            embed_dim=128
        ).to(self.device)

    def forward(self, anomaly_data, use_casual=True):

        gat_Origin_F = self.gat_encoder(anomaly_data.to(self.device))

        batch_size, n_heads, _ = gat_Origin_F.shape

        if use_casual:
            Origin_F_reshaped = pca_feature_with_mean_replacement(gat_Origin_F.view(batch_size, -1),mask_type='mean')
            gat_C_F = torch.tensor(Origin_F_reshaped, device=self.device).view(batch_size, n_heads, self.gat_out_dim)

            C_F = self.cross_attention(gat_Origin_F, gat_C_F).view(batch_size, -1)
        else:
            C_F = self.cross_attention(gat_Origin_F, gat_Origin_F).view(batch_size, -1)

        return C_F

    def get_pseudo_label(self,feature):
        pseudo_label = self.Decoder(feature)
        return pseudo_label


class CrossAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, out_dim, dropout=0.2):
        super(CrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.linear_layer = nn.Linear(input_dim, self.embed_dim)
        self.out_layer = nn.Linear(input_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.scale = nn.Parameter(torch.ones(out_dim))
        self.fc_out = nn.Linear(embed_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, T, S):
        # Compute Q, K, V
        KV = self.linear_layer(S.mean(dim=1))
        Q = T.mean(dim=1)
        Q = self.linear_layer(Q)

        # Compute attention scores
        attn_scores = torch.matmul(Q, KV.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute weighted sum of V
        output = torch.matmul(attn_weights, KV)

        # Apply the rest of the operations
        output = self.dropout(self.fc_out(output))
        output = self.layer_norm(output) * self.scale + self.out_layer(T.mean(dim=1))

        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()
        layers = []
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = dim
        layers.extend([nn.Linear(hidden_dims[-1],1)])
        self.decoder = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        score = self.sigmoid(self.decoder(x))
        return score

def compute_mmd(x, y, kernel_fn):
    """
    Compute maximum mean discrepancy (MMD) between samples x and y.
    """
    xx = kernel_fn(x, x)
    yy = kernel_fn(y, y)
    xy = kernel_fn(x, y)
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


def gaussian_kernel(a, b):
    """
    Compute Gaussian kernel between samples a and b.
    """
    squared_distance = ((a - b) ** 2).sum(-1)
    return torch.exp(-squared_distance / 2)
